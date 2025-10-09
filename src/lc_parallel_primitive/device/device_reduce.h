/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:52:20
 */

#pragma once
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
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>
#include <lc_parallel_primitive/block/block_reduce.h>
#include <lc_parallel_primitive/warp/warp_reduce.h>
#include <typeindex>
namespace luisa::parallel_primitive
{

using namespace luisa::compute;
class DeviceReduce : public LuisaModule
{
  private:
    template <NumericT Type4Byte>
    using IndexValuePairT = KeyValuePair<int, Type4Byte>;

    template <NumericT Type4Byte>
    using ReduceShaderT =
        Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int, Type4Byte>;

    template <NumericT Type4Byte>
    using ArgReduceShaderT =
        Shader<1, Buffer<IndexValuePairT<Type4Byte>>, Buffer<IndexValuePairT<Type4Byte>>, int, int, int, IndexValuePairT<Type4Byte>>;

    template <NumericT Type4Byte>
    using ArgConstructShaderT =
        Shader<1, Buffer<Type4Byte>, Buffer<IndexValuePairT<Type4Byte>>>;

    template <NumericT Type4Byte>
    using ArgAssignShaderT =
        Shader<1, Buffer<IndexValuePairT<Type4Byte>>, Buffer<Type4Byte>, Buffer<int>>;

    // for reduce by key
    template <NumericT KeyType, NumericT ValueType>
    using KVPT = KeyValuePair<KeyType, ValueType>;

    template <NumericT KeyType, NumericT ValueType>
    using ConstructKVPShaderT =
        Shader<1, Buffer<KeyType>, Buffer<ValueType>, Buffer<KVPT<KeyType, ValueType>>>;

    template <NumericT KeyType, NumericT ValueType>
    using AssignKVPShaderT =
        Shader<1, Buffer<KVPT<KeyType, ValueType>>, Buffer<KeyType>, Buffer<ValueType>>;

  public:
    int m_block_size = 256;
    int m_warp_nums  = 32;

  private:
    int    m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * 2;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        compile_common<int>(device);
        compile_common<float>(device);
        m_created = true;
    }

    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                ReduceOp              op,
                Type4Byte             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, op, init);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<IndexValuePairT<Type4Byte>>(
                  const Var<IndexValuePairT<Type4Byte>>&, Var<IndexValuePairT<Type4Byte>>&)>>
    void ArgReduce(CommandList&                           cmdlist,
                   Stream&                                stream,
                   BufferView<IndexValuePairT<Type4Byte>> d_in,
                   BufferView<IndexValuePairT<Type4Byte>> d_out,
                   size_t                                 num_item,
                   ReduceOp                               op,
                   IndexValuePairT<Type4Byte>             init)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        Buffer<IndexValuePairT<Type4Byte>> temp_buffer =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(temp_storage_size);
        arg_reduce_array_recursive<Type4Byte>(
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
                BufferView<int>       d_index_out,
                size_t                num_item)
    {
        // key value pair reduce
        Buffer<IndexValuePairT<Type4Byte>> d_in_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_kvp_construct_it = ms_arg_construct_map.find(key);
        auto ms_kvp_construct_ptr = reinterpret_cast<ArgConstructShaderT<Type4Byte>*>(
            &(*ms_kvp_construct_it->second));
        cmdlist << (*ms_kvp_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());

        Buffer<IndexValuePairT<Type4Byte>> d_out_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);

        ArgReduce(
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
        auto ms_kvp_assign_ptr =
            reinterpret_cast<ArgAssignShaderT<Type4Byte>*>(&(*ms_kvp_assign_it->second));
        cmdlist << (*ms_kvp_assign_ptr)(d_out_kv.view(), d_out, d_index_out)
                       .dispatch(d_index_out.size());
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<int>       d_index_out,
                size_t                num_item)
    {
        // key value pair reduce
        Buffer<IndexValuePairT<Type4Byte>> d_in_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShaderT<Type4Byte>*>(
            &(*ms_arg_construct_it->second));
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());

        Buffer<IndexValuePairT<Type4Byte>> d_out_kv =
            m_device.create_buffer<IndexValuePairT<Type4Byte>>(1);


        ArgReduce(
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
        auto ms_arg_assign_ptr =
            reinterpret_cast<ArgAssignShaderT<Type4Byte>*>(&(*ms_arg_assign_it->second));
        cmdlist << (*ms_arg_assign_ptr)(d_out_kv.view(), d_out, d_index_out)
                       .dispatch(d_index_out.size());
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT KeyType,
              NumericT ValueType,
              typename ReduceOp = luisa::compute::Callable<Var<ValueType>(const Var<ValueType>&, Var<ValueType>&)>>
    void ReduceByKey(CommandList&          cmdlist,
                     Stream&               stream,
                     BufferView<KeyType>   d_keys_in,
                     BufferView<ValueType> d_values_in,
                     BufferView<KeyType>   d_unique_out,
                     BufferView<ValueType> d_aggregates_out,
                     BufferView<int>       d_num_runs_out,
                     ReduceOp              reduce_op = 0)
    {
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
        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;
        // for construct key value pair
        luisa::unique_ptr<ArgConstructShaderT<Type4Byte>> ms_arg_construct_shader = nullptr;
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


        luisa::unique_ptr<ArgAssignShaderT<Type4Byte>> ms_arg_assign_shader = nullptr;
        lazy_compile(device,
                     ms_arg_assign_shader,
                     [&](BufferVar<IndexValuePairT<Type4Byte>> kvp_in,
                         BufferVar<Type4Byte>                  value_out,
                         BufferVar<int> index_out) noexcept
                     {
                         set_block_size(m_block_size);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);

                         Var<IndexValuePairT<Type4Byte>> kvp = kvp_in.read(global_id);
                         index_out.write(global_id, kvp.key);
                         value_out.write(global_id, kvp.value);
                     });
        ms_arg_assign_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_arg_assign_shader));
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void compile_reduce_op(Device& device, ReduceOp reduce_op)
    {
        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;

        // for reduce
        auto load_shared_chunk_from_mem = [&](SmemTypePtr<Type4Byte>& s_data,
                                              BufferVar<Type4Byte>&   g_idata,
                                              Int                     n,
                                              Int                     baseIndex,
                                              Var<Type4Byte>          initial)
        {
            Int thread_id_x = Int(thread_id().x);
            Int block_id_x  = Int(block_id().x);
            Int block_dim_x = Int(block_size_x());

            Int thid   = thread_id_x;
            Int men_ai = baseIndex + thread_id_x;
            Int men_bi = men_ai + block_dim_x;

            Int ai = thid;
            Int bi = thid + block_dim_x;  // bank conflict free

            Int bank_offset_a = conflict_free_offset(ai);
            Int bank_offset_b = conflict_free_offset(bi);

            Var<Type4Byte> data_ai = initial;
            Var<Type4Byte> data_bi = initial;

            $if(ai < n)
            {
                data_ai = g_idata.read(men_ai);
            };
            $if(bi < n)
            {
                data_bi = g_idata.read(men_bi);
            };
            (*s_data)[ai + bank_offset_a] = data_ai;
            (*s_data)[bi + bank_offset_b] = data_bi;
        };

        auto reduce_block = [&](SmemTypePtr<Type4Byte>& s_data,
                                BufferVar<Type4Byte>&   block_sums,
                                Int                     block_index,
                                Var<Type4Byte>          initial)
        {
            $if(block_index == 0)
            {
                block_index = Int(block_id().x);
            };
            // build the op in place up the tree
            Int thid   = Int(thread_id().x);
            Int stride = def(1);

            // build the sum in place up the tree
            Int d = Int(block_size().x);
            $while(d > 0)
            {
                sync_block();
                $if(thid < d)
                {
                    Int i  = (stride * 2) * thid;
                    Int ai = i + stride - 1;
                    Int bi = ai + stride;
                    ai += conflict_free_offset(ai);
                    bi += conflict_free_offset(bi);
                    (*s_data)[bi] = reduce_op((*s_data)[bi], (*s_data)[ai]);
                };
                stride *= 2;
                d = d >> 1;
            };
            d = Int(block_size_x());
            $if(thid == 0)
            {
                Int index = (d << 1) - 1;
                index += conflict_free_offset(index);
                block_sums.write(block_index, (*s_data)[index]);
                (*s_data)[index] = Type4Byte(0);
            };
        };
        //  reduce
        luisa::unique_ptr<ReduceShaderT<Type4Byte>> ms_reduce_shader = nullptr;
        lazy_compile(device,
                     ms_reduce_shader,
                     [&](BufferVar<Type4Byte> g_idata,
                         BufferVar<Type4Byte> g_block_sums,
                         Int                  n,
                         Int                  block_index,
                         Int                  base_index,
                         Var<Type4Byte>       init) noexcept
                     {
                         set_block_size(n_blocks);
                         Int ai, bi, men_ai, men_bi, bank_offset_a, bank_offset_b;
                         Int block_id_x  = Int(block_id().x);
                         Int block_dim_x = Int(block_size_x());
                         SmemTypePtr<Type4Byte> s_data =
                             new SmemType<Type4Byte>{shared_mem_size};

                         $if(base_index == 0)
                         {
                             base_index = block_id_x * (block_dim_x << 1);
                         };
                         load_shared_chunk_from_mem(s_data, g_idata, n, base_index, init);
                         reduce_block(s_data, g_block_sums, block_index, n);
                     });

        ms_reduce_map.try_emplace(get_reduce_type_op_desc<Type4Byte>(reduce_op),
                                  std::move(ms_reduce_shader));
    }

    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<IndexValuePairT<Type4Byte>>(
                  const Var<IndexValuePairT<Type4Byte>>&, Var<IndexValuePairT<Type4Byte>>&)>>
    void compile_arg_reduce_op(Device& device, ReduceOp reduce_op)
    {
        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;

        auto load_shared_chunk_from_mem_op_arg =
            [&](SmemTypePtr<IndexValuePairT<Type4Byte>>& s_data,
                BufferVar<IndexValuePairT<Type4Byte>>&   g_idata,
                Int                                      n,
                Int&                                     baseIndex,
                Var<IndexValuePairT<Type4Byte>>          initial)
        {
            Int thread_id_x = Int(thread_id().x);
            Int block_id_x  = Int(block_id().x);
            Int block_dim_x = Int(block_size_x());

            Int thid   = thread_id_x;
            Int men_ai = baseIndex + thread_id_x;
            Int men_bi = men_ai + block_dim_x;

            Int ai = thid;
            Int bi = thid + block_dim_x;  // bank conflict free

            Int bank_offset_a = conflict_free_offset(ai);
            Int bank_offset_b = conflict_free_offset(bi);

            Var<IndexValuePairT<Type4Byte>> data_ai = initial;
            Var<IndexValuePairT<Type4Byte>> data_bi = initial;

            $if(ai < n)
            {
                data_ai = g_idata.read(men_ai);
            };
            $if(bi < n)
            {
                data_bi = g_idata.read(men_bi);
            };
            (*s_data)[ai + bank_offset_a] = data_ai;
            (*s_data)[bi + bank_offset_b] = data_bi;
        };

        auto arg_reduce_block = [&](SmemTypePtr<IndexValuePairT<Type4Byte>>& s_data,
                                    BufferVar<IndexValuePairT<Type4Byte>>& block_sums,
                                    Int block_index,
                                    Int n)
        {
            $if(block_index == 0)
            {
                block_index = Int(block_id().x);
            };

            Int thid   = Int(thread_id().x);
            Int stride = def(1);

            // build the sum in place up the tree
            Int d = Int(block_size().x);
            $while(d > 0)
            {
                sync_block();
                $if(thid < d)
                {
                    Int i  = (stride * 2) * thid;
                    Int ai = i + stride - 1;
                    Int bi = ai + stride;
                    ai += conflict_free_offset(ai);
                    bi += conflict_free_offset(bi);
                    (*s_data)[bi] = reduce_op((*s_data)[bi], (*s_data)[ai]);
                };
                stride *= 2;
                d = d >> 1;
            };
            d = Int(block_size_x());
            $if(thid == 0)
            {
                Int index = (d << 1) - 1;
                index += conflict_free_offset(index);
                block_sums.write(block_index, (*s_data)[index]);
                // zero the last element in the scan so it will propagate back to the front
                (*s_data)[index].value = Type4Byte(0);
                (*s_data)[index].key   = 0;
            };
        };
        luisa::unique_ptr<ArgReduceShaderT<Type4Byte>> ms_arg_reduce_shader = nullptr;
        lazy_compile(device,
                     ms_arg_reduce_shader,
                     [&](BufferVar<IndexValuePairT<Type4Byte>> g_idata,
                         BufferVar<IndexValuePairT<Type4Byte>> g_block_sums,
                         Int                                   n,
                         Int                                   block_index,
                         Int                                   base_index,
                         Var<IndexValuePairT<Type4Byte>>       init) noexcept
                     {
                         set_block_size(m_block_size);
                         Int block_id_x  = Int(block_id().x);
                         Int block_dim_x = Int(block_size_x());

                         SmemTypePtr<IndexValuePairT<Type4Byte>> s_data =
                             new SmemType<IndexValuePairT<Type4Byte>>{shared_mem_size};

                         $if(base_index == 0)
                         {
                             base_index = block_id_x * (block_dim_x << 1);
                         };
                         load_shared_chunk_from_mem_op_arg(s_data, g_idata, n, base_index, init);
                         arg_reduce_block(s_data, g_block_sums, block_index, n);
                     });
        ms_arg_reduce_map.try_emplace(get_reduce_type_op_desc<Type4Byte>(reduce_op),
                                      std::move(ms_arg_reduce_shader));
    }


    template <NumericT KeyType, NumericT ValueType>
    void compile_reduce_by_key(Device& device)
    {
        using KVP = KeyValuePair<KeyType, ValueType>;
        // to be implemented
        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;

        // for construct/assign key value pair
        luisa::unique_ptr<ConstructKVPShaderT<KeyType, ValueType>> ms_kvp_construct_shader =
            nullptr;
        lazy_compile(device,
                     ms_kvp_construct_shader,
                     [&](BufferVar<KeyType>   arr_key_in,
                         BufferVar<ValueType> arr_value_in,
                         BufferVar<KVP>       g_kv_out) noexcept
                     {
                         set_block_size(n_blocks);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);
                         Var<KVP> initial{arr_key_in.read(global_id),
                                          arr_value_in.read(global_id)};
                         g_kv_out.write(global_id, initial);
                     });
        ms_kvp_construct_shader.try_emplace(
            {luisa::string{luisa::compute::Type::of<KeyType>()->description()},
             luisa::string{luisa::compute::Type::of<ValueType>()->description()}},
            std::move(ms_kvp_construct_shader));

        luisa::unique_ptr<ConstructKVPShaderT<KeyType, ValueType>> ms_kvp_assign_shader = nullptr;
        lazy_compile(device,
                     ms_kvp_assign_shader,
                     [&](BufferVar<KVP>       g_kv_in,
                         BufferVar<KeyType>   arr_key_out,
                         BufferVar<ValueType> arr_value_in) noexcept
                     {
                         set_block_size(n_blocks);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);
                         Var<KVP> key_value_pair = g_kv_in.read(global_id);
                         arr_key_out.write(global_id, key_value_pair.key);
                         arr_value_in.write(global_id, key_value_pair.value);
                     });
        ms_kvp_assign_map.try_emplace(
            {luisa::string{luisa::compute::Type::of<KeyType>()->description()},
             luisa::string{luisa::compute::Type::of<ValueType>()->description()}},
            std::move(ms_kvp_assign_shader));
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                BufferView<Type4Byte>        temp_storage,
                                BufferView<Type4Byte>        arr_in,
                                BufferView<Type4Byte>        arr_out,
                                int                          num_elements,
                                int                          offset,
                                int                          level,
                                ReduceOp                     op,
                                Type4Byte                    init) noexcept
    {
        int block_size = m_block_size;
        int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
        int num_threads;

        if(num_blocks > 1)
        {
            num_threads = block_size;
        }
        else if(is_power_of_two(num_elements))
        {
            num_threads = num_elements / 2;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * 2;
        int num_elements_last_block = num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block / 2);
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
        size_t                size_elements = temp_storage.size() - offset;
        BufferView<Type4Byte> temp_buffer_level =
            temp_storage.subview(offset, size_elements);
        // execute the scan
        auto key          = get_reduce_type_op_desc<Type4Byte>(op);
        auto ms_reduce_it = ms_reduce_map.find(key);
        if(ms_reduce_it == ms_reduce_map.end())
        {
            compile_reduce_op<Type4Byte>(m_device, op);
            ms_reduce_it = ms_reduce_map.find(key);
        }
        auto ms_reduce_ptr =
            reinterpret_cast<ReduceShaderT<Type4Byte>*>(&(*ms_reduce_it->second));

        if(num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(arr_in,
                                            temp_buffer_level,
                                            num_elements_last_block,
                                            num_blocks - 1,
                                            num_elements - num_elements_last_block,
                                            init)
                               .dispatch(block_size);
            }

            reduce_array_recursive<Type4Byte>(cmdlist,
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
                           .dispatch(block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };


    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<IndexValuePairT<Type4Byte>>(
                  const Var<IndexValuePairT<Type4Byte>>&, const Var<IndexValuePairT<Type4Byte>>&)>>
    void arg_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                    BufferView<IndexValuePairT<Type4Byte>> temp_storage,
                                    BufferView<IndexValuePairT<Type4Byte>> arr_in,
                                    BufferView<IndexValuePairT<Type4Byte>> arr_out,
                                    int                        num_elements,
                                    int                        offset,
                                    int                        level,
                                    ReduceOp                   op,
                                    IndexValuePairT<Type4Byte> init) noexcept
    {
        int block_size = m_block_size;
        int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
        int num_threads;

        if(num_blocks > 1)
        {
            num_threads = block_size;
        }
        else if(is_power_of_two(num_elements))
        {
            num_threads = num_elements / 2;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * 2;
        int num_elements_last_block = num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block / 2);
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
        BufferView<IndexValuePairT<Type4Byte>> temp_buffer_level =
            temp_storage.subview(offset, size_elements);
        // execute the scan
        auto key              = get_reduce_type_op_desc<Type4Byte>(op);
        auto ms_arg_reduce_it = ms_arg_reduce_map.find(key);
        if(ms_arg_reduce_it == ms_arg_reduce_map.end())
        {
            compile_arg_reduce_op<Type4Byte>(m_device, op);
            ms_arg_reduce_it = ms_arg_reduce_map.find(key);
        }
        auto ms_arg_reduce_ptr =
            reinterpret_cast<ArgReduceShaderT<Type4Byte>*>(&(*ms_arg_reduce_it->second));

        if(num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_arg_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_arg_reduce_ptr)(arr_in,
                                                temp_buffer_level,
                                                num_elements_last_block,
                                                num_blocks - 1,
                                                num_elements - num_elements_last_block,
                                                init)
                               .dispatch(block_size);
            }

            arg_reduce_array_recursive<Type4Byte>(cmdlist,
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
            cmdlist << (*ms_arg_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, init)
                           .dispatch(block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };

    template <NumericT KeyType, NumericT ValueType>
    void reduce_by_key_array_recursive(luisa::compute::CommandList& cmdlist,
                                       int num_elements,
                                       int offset,
                                       int level,
                                       int op)
    {
        int block_size = m_block_size;
        int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
        int num_threads;

        if(num_blocks > 1)
        {
            num_threads = block_size;
        }
        else if(is_power_of_two(num_elements))
        {
            num_threads = num_elements / 2;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * 2;
        int num_elements_last_block = num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block / 2);
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
    };


    template <NumericT Type4Btye, typename ReduceOp>
    luisa::string get_reduce_type_op_desc(ReduceOp op)
    {
        luisa::string_view key_desc =
            luisa::compute::Type::of<Type4Btye>()->description();
        luisa::string_view reduce_op_desc = std::type_index(typeid(op)).name();

        return luisa::string(key_desc) + "+" + luisa::string(reduce_op_desc);
    }


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    // for arg reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_reduce_map;
    // for reduce by key
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_kvp_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_kvp_assign_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;
};
}  // namespace luisa::parallel_primitive