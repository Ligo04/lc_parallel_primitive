/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 18:31:46
 */

#pragma once
#include "luisa/core/mathematics.h"
#include "luisa/dsl/func.h"
#include <luisa/dsl/local.h>
#include <limits>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/device_reduce.h>
#include <lcpp/device/details/device_reduce_by_key.h>
#include <typeindex>
namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = 256, size_t ITEMS_PER_THREAD = 1, size_t WARP_NUMS = 32>
class DeviceReduce : public LuisaModule
{
  private:
    template <NumericT Type4Byte>
    using IndexValuePairT = details::IndexValuePairT<Type4Byte>;

  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        compile_common<int>(device);
        compile_common<float>(device);
        m_created = true;
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                ReduceOp              op,
                Type4Byte             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, op, init);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&                           cmdlist,
                Stream&                                stream,
                BufferView<IndexValuePairT<Type4Byte>> d_in,
                BufferView<IndexValuePairT<Type4Byte>> d_out,
                size_t                                 num_item,
                ReduceOp                               op,
                IndexValuePairT<Type4Byte>             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<IndexValuePairT<Type4Byte>> temp_buffer =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(temp_storage_size);
        reduce_array_recursive<IndexValuePairT<Type4Byte>>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, op, init);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b) { return a + b; },
            0);
    }

    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b)
            { return luisa::compute::min(a, b); },
            std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b)
            { return luisa::compute::max(a, b); },
            std::numeric_limits<Type4Byte>::min());
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        // key value pair reduce
        Buffer<IndexValuePairT<Type4Byte>> d_in_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_kvp_construct_it = ms_arg_construct_map.find(key);
        auto ms_kvp_construct_ptr =
            reinterpret_cast<details::ArgConstructShaderT<Type4Byte>*>(
                &(*ms_kvp_construct_it->second));
        cmdlist << (*ms_kvp_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());

        Buffer<IndexValuePairT<Type4Byte>> d_out_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);

        Reduce(
            cmdlist,
            stream,
            d_in_kv.view(),
            d_out_kv.view(),
            num_item,
            [](const Var<IndexValuePairT<Type4Byte>>& a, Var<IndexValuePairT<Type4Byte>>& b)
            {
                Var<IndexValuePairT<Type4Byte>> result = a;
                $if(b.value < a.value | (b.value == a.value & b.key < a.key))
                {
                    result = b;
                };
                return result;
            },
            IndexValuePairT<Type4Byte>{0, std::numeric_limits<Type4Byte>::max()});

        // copy result to d_out and d_index_out
        auto ms_kvp_assign_it = ms_arg_assign_map.find(key);
        auto ms_kvp_assign_ptr = reinterpret_cast<details::ArgAssignShaderT<Type4Byte>*>(
            &(*ms_kvp_assign_it->second));
        cmdlist << (*ms_kvp_assign_ptr)(d_out_kv.view(), d_out, d_index_out)
                       .dispatch(d_index_out.size());
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        // key value pair reduce
        Buffer<IndexValuePairT<Type4Byte>> d_in_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        auto ms_arg_construct_ptr =
            reinterpret_cast<details::ArgConstructShaderT<Type4Byte>*>(
                &(*ms_arg_construct_it->second));
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());
        Buffer<IndexValuePairT<Type4Byte>> d_out_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);

        Reduce(
            cmdlist,
            stream,
            d_in_kv.view(),
            d_out_kv.view(),
            num_item,
            [](const Var<IndexValuePairT<Type4Byte>>& a, Var<IndexValuePairT<Type4Byte>>& b)
            {
                Var<IndexValuePairT<Type4Byte>> result = a;
                $if(b.value > a.value | (b.value == a.value & b.key > a.key))
                {
                    result = b;
                };
                return result;
            },
            IndexValuePairT<Type4Byte>{0, std::numeric_limits<Type4Byte>::min()});


        // copy result to d_out and d_index_out
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        auto ms_arg_assign_ptr = reinterpret_cast<details::ArgAssignShaderT<Type4Byte>*>(
            &(*ms_arg_assign_it->second));
        cmdlist << (*ms_arg_assign_ptr)(d_out_kv.view(), d_out, d_index_out)
                       .dispatch(d_index_out.size());
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT KeyType,
              NumericT ValueType,
              typename ReduceOp = luisa::compute::Callable<Var<ValueType>(const Var<ValueType>&, const Var<ValueType>&)>>
    void ReduceByKey(CommandList&          cmdlist,
                     Stream&               stream,
                     BufferView<KeyType>   d_keys_in,
                     BufferView<ValueType> d_values_in,
                     BufferView<KeyType>   d_unique_out,
                     BufferView<ValueType> d_aggregates_out,
                     BufferView<uint>      d_num_runs_out,
                     ReduceOp              reduce_op,
                     size_t                num_element)
    {
        reduce_by_key_array(
            cmdlist, d_keys_in, d_values_in, d_unique_out, d_aggregates_out, d_num_runs_out, reduce_op, num_element);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>,
              typename TransfromOp>
    void TransformReduce(CommandList&          cmdlist,
                         BufferView<Type4Byte> temp_buffer,
                         BufferView<Type4Byte> d_in,
                         BufferView<Type4Byte> d_out,
                         size_t                num_item,
                         ReduceOp              reduce_op,
                         TransfromOp           transform_op,
                         Type4Byte             init)
    {
    }

  private:
    template <NumericT Type4Byte>
    void compile_common(Device& device)
    {
        const auto   n_blocks        = m_block_size;
        const size_t shared_mem_size = m_shared_mem_size;

        // for construct key value pair
        luisa::unique_ptr<details::ArgConstructShaderT<Type4Byte>> ms_arg_construct_shader =
            nullptr;
        lazy_compile(device,
                     ms_arg_construct_shader,
                     [&](BufferVar<Type4Byte> arr_in,
                         BufferVar<IndexValuePairT<Type4Byte>> g_kv_out) noexcept
                     {
                         set_block_size(m_block_size);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);

                         Var<IndexValuePairT<Type4Byte>> initial{0, 0};
                         initial.key   = global_id;
                         initial.value = arr_in.read(global_id);
                         g_kv_out.write(global_id, initial);
                     });
        ms_arg_construct_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_arg_construct_shader));


        luisa::unique_ptr<details::ArgAssignShaderT<Type4Byte>> ms_arg_assign_shader = nullptr;
        lazy_compile(device,
                     ms_arg_assign_shader,
                     [&](BufferVar<IndexValuePairT<Type4Byte>> kvp_in,
                         BufferVar<Type4Byte>                  value_out,
                         BufferVar<uint> index_out) noexcept
                     {
                         set_block_size(m_block_size);
                         UInt global_id =
                             UInt(block_id().x * block_size().x + thread_id().x);

                         Var<IndexValuePairT<Type4Byte>> kvp = kvp_in.read(global_id);
                         index_out.write(global_id, kvp.key);
                         value_out.write(global_id, kvp.value);
                     });
        ms_arg_assign_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_arg_assign_shader));
    }

    template <details::NumericTOrKeyValuePairT Type, typename ReduceOp>
    void reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                BufferView<Type>             temp_storage,
                                BufferView<Type>             arr_in,
                                BufferView<Type>             arr_out,
                                int                          num_elements,
                                int                          offset,
                                int                          level,
                                ReduceOp                     op,
                                Type                         init) noexcept
    {
        int num_blocks =
            imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        int num_threads;

        if(num_blocks > 1)
        {
            num_threads = m_block_size;
        }
        else if(is_power_of_two(num_elements))
        {
            num_threads = num_elements / ITEMS_PER_THREAD;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * ITEMS_PER_THREAD;
        int num_elements_last_block = num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block);
        int np2_last_block         = 0;
        int shared_mem_last_block  = 0;

        if(num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if(!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }
        size_t size_elements = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);
        auto key          = get_reduce_type_op_desc<Type>(op);
        auto ms_reduce_it = ms_reduce_map.find(key);
        if(ms_reduce_it == ms_reduce_map.end())
        {
            auto shader =
                details::ReduceShader<Type, BLOCK_SIZE, ITEMS_PER_THREAD>().compile(
                    m_device, m_shared_mem_size, op);
            ms_reduce_map.try_emplace(key, std::move(shader));
            ms_reduce_it = ms_reduce_map.find(key);
        }
        auto ms_reduce_ptr =
            reinterpret_cast<details::ReduceShaderT<Type>*>(&(*ms_reduce_it->second));

        if(num_blocks > 1)
        {
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(m_block_size * (num_blocks - np2_last_block));

            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(arr_in,
                                            temp_buffer_level,
                                            num_elements_last_block,
                                            num_blocks - 1,
                                            num_elements - num_elements_last_block,
                                            init)
                               .dispatch(m_block_size);
            }
            // recursive
            reduce_array_recursive<Type>(cmdlist,
                                         temp_buffer_level,
                                         temp_buffer_level,
                                         arr_out,
                                         num_blocks,
                                         num_blocks,
                                         level + 1,
                                         op,
                                         init);
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(m_block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };

    template <NumericT KeyType, NumericT ValueType, typename ReduceOp>
    void reduce_by_key_array(CommandList&          cmdlist,
                             BufferView<KeyType>   keys_in,
                             BufferView<ValueType> values_in,
                             BufferView<KeyType>   unique_out,
                             BufferView<ValueType> aggregated_out,
                             BufferView<uint>      num_runs_out,
                             ReduceOp              reduce_op,
                             int                   num_elements) noexcept
    {
        int num_blocks =
            imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        int num_threads;

        if(num_blocks > 1)
        {
            num_threads = m_block_size;
        }
        else if(is_power_of_two(num_elements))
        {
            num_threads = num_elements / ITEMS_PER_THREAD;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * ITEMS_PER_THREAD;
        int num_elements_last_block = num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block);
        int np2_last_block         = 0;
        int shared_mem_last_block  = 0;

        if(num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if(!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }

        // Initialize output counter
        std::vector<uint> zero_data(1, 0);
        cmdlist << num_runs_out.copy_from(zero_data.data());
        // execute the scan
        auto key = get_key_value_op_shader_desc<KeyType, ValueType>(reduce_op);
        auto ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        if(ms_reduce_by_key_it == ms_reduce_by_key_map.end())
        {
            auto shader =
                details::ReduceByKeyShader<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>()
                    .compile(m_device, m_shared_mem_size, reduce_op);
            ms_reduce_by_key_map.try_emplace(key, std::move(shader));
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr =
            reinterpret_cast<details::ReduceByKeyShaderT<KeyType, ValueType>*>(
                &(*ms_reduce_by_key_it->second));

        cmdlist << (*ms_reduce_by_key_ptr)(
                       keys_in, values_in, unique_out, aggregated_out, num_runs_out, num_elements)
                       .dispatch(m_block_size * num_blocks);
    };


    template <NumericT Type4Byte, typename ReduceOp>
    luisa::string get_reduce_type_op_desc(ReduceOp op)
    {
        luisa::string_view key_desc =
            luisa::compute::Type::of<Type4Byte>()->description();
        luisa::string_view reduce_op_desc = std::type_index(typeid(op)).name();

        return luisa::string(key_desc) + "+" + luisa::string(reduce_op_desc);
    }

    template <KeyValuePairType KeyValueType, typename ReduceOp>
    luisa::string get_reduce_type_op_desc(ReduceOp op)
    {
        using ValueType = value_type_of_t<KeyValueType>;
        luisa::string_view key_desc =
            luisa::compute::Type::of<ValueType>()->description();
        luisa::string_view reduce_op_desc = std::type_index(typeid(op)).name();

        return luisa::string(key_desc) + "+" + luisa::string(reduce_op_desc);
    }


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    // for arg reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    // for reduce by key
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_partial_reduce_by_key_map;
};
}  // namespace luisa::parallel_primitive