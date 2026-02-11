/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 17:31:05
 */

#pragma once
#include <luisa/core/mathematics.h>
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
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/reduce.h>
#include <lcpp/device/details/reduce_by_key.h>
namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceReduce : public LuisaModule
{
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
        m_created                  = true;
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_array_recursive<Type4Byte>(cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, reduce_op, initial_value);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&                           cmdlist,
                Stream&                                stream,
                BufferView<IndexValuePairT<Type4Byte>> d_in,
                BufferView<IndexValuePairT<Type4Byte>> d_out,
                size_t                                 num_item,
                ReduceOp                               reduce_op,
                IndexValuePairT<Type4Byte>             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<IndexValuePairT<Type4Byte>> temp_buffer =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(temp_storage_size);
        reduce_array_recursive<IndexValuePairT<Type4Byte>>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, reduce_op, init);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte>
    void Sum(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void Min(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b) { return luisa::compute::min(a, b); },
            std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_item,
            [](const Var<Type4Byte>& a, Var<Type4Byte>& b) { return luisa::compute::max(a, b); },
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

        Buffer<IndexValuePairT<Type4Byte>> d_out_kv = m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);

        // construct key value pair
        arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv.view());

        Reduce(cmdlist,
               stream,
               d_in_kv.view(),
               d_out_kv.view(),
               num_item,
               ArgMinOp(),
               IndexValuePairT<Type4Byte>{0, std::numeric_limits<Type4Byte>::max()});

        // copy result to d_out and d_index_out
        arg_assign<Type4Byte>(cmdlist, d_out_kv.view(), d_out, d_index_out);

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
        Buffer<IndexValuePairT<Type4Byte>> d_out_kv = m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);

        // construct key value pair
        arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv.view());

        Reduce(cmdlist,
               stream,
               d_in_kv.view(),
               d_out_kv.view(),
               num_item,
               ArgMaxOp(),
               IndexValuePairT<Type4Byte>{0, std::numeric_limits<Type4Byte>::min()});


        // copy result to d_out and d_index_out
        arg_assign<Type4Byte>(cmdlist, d_out_kv.view(), d_out, d_index_out);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT KeyType, NumericT ValueType, typename ReduceOp>
    void ReduceByKey(CommandList&          cmdlist,
                     Stream&               stream,
                     BufferView<KeyType>   d_keys_in,
                     BufferView<ValueType> d_values_in,
                     BufferView<KeyType>   d_unique_out,
                     BufferView<ValueType> g_aggregates_out,
                     BufferView<uint>      g_num_runs_out,
                     ReduceOp              reduce_op,
                     size_t                num_elements)
    {
        luisa::vector<luisa::uint> zero_data(1, 0);
        stream << g_num_runs_out.copy_from(zero_data.data()) << synchronize();
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        // tilestate
        using ReduceByKey = details::ReduceByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceByKeyTileState = ReduceByKey::ScanTileState;
        Buffer<ReduceByKeyTileState> tile_states =
            m_device.create_buffer<ReduceByKeyTileState>(details::WARP_SIZE + num_tiles);

        reduce_by_key_array<KeyType, ValueType, ReduceByKeyTileState>(
            cmdlist, tile_states.view(), d_keys_in, d_values_in, d_unique_out, g_aggregates_out, g_num_runs_out, reduce_op, num_elements);
        stream << cmdlist.commit() << synchronize();
    }


    template <typename Type4Byte, typename ReduceOp, typename TransformOp>
    void TransformReduce(CommandList&          cmdlist,
                         Stream&               stream,
                         BufferView<Type4Byte> d_in,
                         BufferView<Type4Byte> d_out,
                         size_t                num_item,
                         ReduceOp              reduce_op,
                         TransformOp           transform_op,
                         Type4Byte             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_transform_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, reduce_op, transform_op, init);
        stream << cmdlist.commit() << synchronize();
    }

  private:
    template <NumericT Type4Byte>
    void arg_construct(CommandList& cmdlist, BufferView<Type4Byte> d_in, BufferView<IndexValuePairT<Type4Byte>> d_kv_out)
    {
        using ArgReduce          = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgConstructShader = ArgReduce::ArgConstructShaderT;
        auto key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            auto shader = ArgReduce().compile_arg_construct_shader(m_device);
            ms_arg_construct_map.try_emplace(key, std::move(shader));
            ms_arg_construct_it = ms_arg_construct_map.find(key);
        }
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShader*>(&(*ms_arg_construct_it->second));
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(d_in.size());
    }

    template <NumericT Type4Byte>
    void arg_assign(CommandList&                           cmdlist,
                    BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                    BufferView<Type4Byte>                  d_value_out,
                    BufferView<uint>                       d_index_out)
    {
        using ArgReduce       = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgAssignShader = ArgReduce::ArgAssignShaderT;
        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_assign_shader(m_device);
            ms_arg_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_assign_map.find(key);
        }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_value_out, d_index_out).dispatch(d_index_out.size());
    }

    template <NumericTOrKeyValuePairT Type, typename ReduceOp>
    void reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                BufferView<Type>             temp_storage,
                                BufferView<Type>             arr_in,
                                BufferView<Type>             arr_out,
                                int                          num_elements,
                                int                          offset,
                                int                          level,
                                ReduceOp                     reduce_op,
                                Type                         init) noexcept
    {
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        int num_threads;

        if(num_tiles > 1)
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

        int num_elements_per_block  = num_threads * ITEMS_PER_THREAD;
        int num_elements_last_block = num_elements - (num_tiles - 1) * num_elements_per_block;
        int num_threads_last_block  = imax(1, num_elements_last_block);
        int np2_last_block          = 0;
        int shared_mem_last_block   = 0;

        if(num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if(!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }
        using ReduceShader = details::ReduceModule<Type, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceKernel = ReduceShader::ReduceShaderKernel;

        size_t           size_elements     = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);

        auto key          = get_type_and_op_desc<Type>(reduce_op);
        auto ms_reduce_it = ms_reduce_map.find(key);
        if(ms_reduce_it == ms_reduce_map.end())
        {
            LUISA_INFO("Compiling Reduce shader for key: {}", key);
            auto shader = ReduceShader().compile(m_device, m_shared_mem_size, reduce_op, IdentityOp());
            ms_reduce_map.try_emplace(key, std::move(shader));
            ms_reduce_it = ms_reduce_map.find(key);
        }
        auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_reduce_it->second));

        if(num_tiles > 1)
        {
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(m_block_size * (num_tiles - np2_last_block));
            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(arr_in,
                                            temp_buffer_level,
                                            num_elements_last_block,
                                            num_tiles - 1,
                                            num_elements - num_elements_last_block,
                                            init)
                               .dispatch(m_block_size);
            }
            // recursive
            reduce_array_recursive<Type>(
                cmdlist, temp_buffer_level, temp_buffer_level, arr_out, num_tiles, num_tiles, level + 1, reduce_op, init);
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init).dispatch(m_block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };


    template <NumericTOrKeyValuePairT Type, typename ReduceOp, typename TransformOp>
    void reduce_transform_array_recursive(luisa::compute::CommandList& cmdlist,
                                          BufferView<Type>             temp_storage,
                                          BufferView<Type>             arr_in,
                                          BufferView<Type>             arr_out,
                                          int                          num_elements,
                                          int                          offset,
                                          int                          level,
                                          ReduceOp                     reduce_op,
                                          TransformOp                  transform_op,
                                          Type                         init) noexcept
    {
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        int num_threads;
        if(num_tiles > 1)
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

        int num_elements_per_block  = num_threads * ITEMS_PER_THREAD;
        int num_elements_last_block = num_elements - (num_tiles - 1) * num_elements_per_block;
        int num_threads_last_block  = imax(1, num_elements_last_block);
        int np2_last_block          = 0;
        int shared_mem_last_block   = 0;

        if(num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if(!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }
        using ReduceShader = details::ReduceModule<Type, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceKernel = ReduceShader::ReduceShaderKernel;

        size_t           size_elements     = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);

        if(num_tiles > 1)
        {
            auto key          = get_type_and_op_desc<Type>(reduce_op, transform_op);
            auto ms_transform = ms_transform_reduce_map.find(key);
            if(ms_transform == ms_transform_reduce_map.end())
            {
                auto shader = ReduceShader().compile(m_device, m_shared_mem_size, reduce_op, transform_op);
                ms_transform_reduce_map.try_emplace(key, std::move(shader));
                ms_transform = ms_transform_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_transform->second));

            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(m_block_size * (num_tiles - np2_last_block));
            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(arr_in,
                                            temp_buffer_level,
                                            num_elements_last_block,
                                            num_tiles - 1,
                                            num_elements - num_elements_last_block,
                                            init)
                               .dispatch(m_block_size);
            }

            // recursive
            reduce_transform_array_recursive<Type>(
                cmdlist, temp_buffer_level, temp_buffer_level, arr_out, num_tiles, num_tiles, level + 1, reduce_op, transform_op, init);
        }
        else if(level == 0)
        {
            auto key          = get_type_and_op_desc<Type>(reduce_op, transform_op);
            auto ms_transform = ms_transform_reduce_map.find(key);
            if(ms_transform == ms_transform_reduce_map.end())
            {
                auto shader = ReduceShader().compile(m_device, m_shared_mem_size, reduce_op, transform_op);
                ms_transform_reduce_map.try_emplace(key, std::move(shader));
                ms_transform = ms_transform_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_transform->second));
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init).dispatch(m_block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
        else
        {
            auto key          = get_type_and_op_desc<Type>(reduce_op);
            auto ms_reduce_it = ms_reduce_map.find(key);
            if(ms_reduce_it == ms_reduce_map.end())
            {
                auto shader = ReduceShader().compile(m_device, m_shared_mem_size, reduce_op, IdentityOp());
                ms_reduce_map.try_emplace(key, std::move(shader));
                ms_reduce_it = ms_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_reduce_it->second));
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init).dispatch(m_block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };

    template <NumericT KeyType, NumericT ValueType, typename ScanTileState, typename ReduceOp>
    void reduce_by_key_array(luisa::compute::CommandList& cmdlist,
                             BufferView<ScanTileState>    tile_states,
                             BufferView<KeyType>          keys_in,
                             BufferView<ValueType>        values_in,
                             BufferView<KeyType>          unique_out,
                             BufferView<ValueType>        aggregated_out,
                             BufferView<uint>             num_runs_out,
                             ReduceOp                     reduce_op,
                             int                          num_elements) noexcept
    {
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        int num_threads;

        if(num_tiles > 1)
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

        int num_elements_per_block  = num_threads * ITEMS_PER_THREAD;
        int num_elements_last_block = num_elements - (num_tiles - 1) * num_elements_per_block;
        int num_threads_last_block  = imax(1, num_elements_last_block);
        int np2_last_block          = 0;
        int shared_mem_last_block   = 0;
        if(num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if(!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }

        using ReduceByKey = details::ReduceByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceByKeyTileState           = ReduceByKey::ScanTileState;
        using ReduceByKeyTileStateInitKernel = ReduceByKey::ScanTileStateInitKernel;
        using ReduceByKeyKernel              = ReduceByKey::ReduceByKeyKernel;

        // init
        auto init_key                   = get_type_and_op_desc<KeyType, ValueType>();
        auto ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        if(ms_scan_tile_state_init_it == ms_scan_tile_state_init_map.end())
        {
            auto shader = ReduceByKey().compile_scan_tile_state_init(m_device);
            ms_scan_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ReduceByKeyTileStateInitKernel*>(&(*ms_scan_tile_state_init_it->second));
        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, num_tiles).dispatch(num_tiles * m_block_size);
        // reduce by key

        auto key                 = get_type_and_op_desc<KeyType, ValueType>(reduce_op);
        auto ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        if(ms_reduce_by_key_it == ms_reduce_by_key_map.end())
        {
            LUISA_INFO("Compiling ReduceByKey shader for key: {}", key);
            auto shader = ReduceByKey().compile(m_device, m_shared_mem_size, reduce_op);
            ms_reduce_by_key_map.try_emplace(key, std::move(shader));
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr = reinterpret_cast<ReduceByKeyKernel*>(&(*ms_reduce_by_key_it->second));

        cmdlist << (*ms_reduce_by_key_ptr)(tile_states, keys_in, values_in, unique_out, aggregated_out, num_runs_out, num_elements)
                       .dispatch(m_block_size * num_tiles);
    };


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_transform_reduce_map;
    // for arg reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    // for reduce by key
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_tile_state_init_map;
};
}  // namespace luisa::parallel_primitive