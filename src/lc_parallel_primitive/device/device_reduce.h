/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-15 11:55:17
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
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>
#include <lc_parallel_primitive/block/block_reduce.h>
#include <lc_parallel_primitive/block/block_scan.h>
#include <lc_parallel_primitive/block/block_load.h>
#include <lc_parallel_primitive/block/block_store.h>
#include <lc_parallel_primitive/block/block_discontinuity.h>
#include <lc_parallel_primitive/warp/warp_reduce.h>
#include <typeindex>
#include <vector>
namespace luisa::parallel_primitive
{
template <typename T>
concept NumericTOrKeyValuePairT = NumericT<T> || KeyValuePairType<T>;

using namespace luisa::compute;
template <size_t BLOCK_SIZE = 256, size_t ITEMS_PER_THREAD = 1, size_t WARP_NUMS = 32>
class DeviceReduce : public LuisaModule
{
  private:
    template <NumericT Type4Byte>
    using IndexValuePairT = KeyValuePair<uint, Type4Byte>;

    template <NumericTOrKeyValuePairT Type>
    using ReduceShaderT = Shader<1, Buffer<Type>, Buffer<Type>, int, int, int, Type>;

    template <NumericT Type4Byte>
    using ArgConstructShaderT =
        Shader<1, Buffer<Type4Byte>, Buffer<IndexValuePairT<Type4Byte>>>;

    template <NumericT Type4Byte>
    using ArgAssignShaderT =
        Shader<1, Buffer<IndexValuePairT<Type4Byte>>, Buffer<Type4Byte>, Buffer<uint>>;

    template <NumericT KeyType, NumericT ValueType>
    using ReduceByKeyShaderT =
        Shader<1, Buffer<KeyType>, Buffer<ValueType>, Buffer<KeyType>, Buffer<ValueType>, Buffer<uint>, uint>;

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
        get_temp_size_scan(temp_storage_size, m_block_size, ITEMS_PER_THREAD, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, op, init);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte,
              typename ReduceOp = luisa::compute::Callable<Var<IndexValuePairT<Type4Byte>>(
                  const Var<IndexValuePairT<Type4Byte>>&, Var<IndexValuePairT<Type4Byte>>&)>>
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
        auto ms_kvp_construct_ptr = reinterpret_cast<ArgConstructShaderT<Type4Byte>*>(
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
                BufferView<uint>      d_index_out,
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
        auto ms_arg_assign_ptr =
            reinterpret_cast<ArgAssignShaderT<Type4Byte>*>(&(*ms_arg_assign_it->second));
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

    template <NumericTOrKeyValuePairT DataType, typename ReduceOp>
    void compile_reduce_op(Device& device, ReduceOp reduce_op)
    {
        const uint n_blocks        = m_block_size;
        const uint shared_mem_size = m_shared_mem_size;

        // for reduce
        auto load_shared_chunk_from_mem = [&](SmemTypePtr<DataType>& s_data,
                                              BufferVar<DataType>&   g_idata,
                                              Int                    n,
                                              Int                    baseIndex,
                                              Var<DataType>          initial)
        {
            Int thread_id_x = Int(thread_id().x);
            Int block_id_x  = Int(block_id().x);
            Int block_dim_x = Int(block_size_x());

            $for(item, 0u, UInt(ITEMS_PER_THREAD))
            {
                Int shared_idx = thread_id_x + item * block_dim_x;  // bank conflict free
                Int           global_idx = baseIndex + shared_idx;
                Var<DataType> data       = initial;
                $if(global_idx < n)
                {
                    data = g_idata.read(global_idx);
                };
                Int bank_offset = conflict_free_offset(shared_idx);
                (*s_data)[shared_idx + bank_offset] = data;
            };
        };

        auto reduce_block =
            [&](SmemTypePtr<DataType>& s_data, BufferVar<DataType>& block_sums, Int block_index)
        {
            $if(block_index == 0)
            {
                block_index = Int(block_id().x);
            };
            // build the op in place up the tree
            Int thid = Int(thread_id().x);

            Int stride = def(1);
            // build the sum in place up the tree
            Int d = Int(block_size_x());
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
                    $if(bi < Int(shared_mem_size) & ai < Int(shared_mem_size))
                    {
                        (*s_data)[bi] = reduce_op((*s_data)[ai], (*s_data)[bi]);
                    };
                };
                stride *= 2;
                d = d >> 1;
            };

            $if(thid == 0)
            {
                Int index = (block_size_x() * UInt(ITEMS_PER_THREAD)) - 1;
                index += conflict_free_offset(index);
                block_sums.write(block_index, (*s_data)[index]);
            };
        };

        //  reduce
        luisa::unique_ptr<ReduceShaderT<DataType>> ms_reduce_shader = nullptr;
        lazy_compile(
            device,
            ms_reduce_shader,
            [&](BufferVar<DataType> g_idata,
                BufferVar<DataType> g_block_sums,
                Int                 num_elements,
                Int                 block_index,
                Int                 base_index,
                Var<DataType>       init) noexcept
            {
                set_block_size(n_blocks);
                Int block_id_x  = Int(block_id().x);
                Int block_dim_x = Int(block_size_x());
                SmemTypePtr<DataType> s_data = new SmemType<DataType>{shared_mem_size};

                $if(base_index == 0)
                {
                    base_index = block_id_x * (block_dim_x * UInt(ITEMS_PER_THREAD));
                };
                load_shared_chunk_from_mem(s_data, g_idata, num_elements, base_index, init);
                reduce_block(s_data, g_block_sums, block_index);
            });

        ms_reduce_map.try_emplace(get_reduce_type_op_desc<DataType>(reduce_op),
                                  std::move(ms_reduce_shader));
    }

    template <NumericTOrKeyValuePairT Type, typename ReduceOp>
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
            compile_reduce_op<Type>(m_device, op);
            ms_reduce_it = ms_reduce_map.find(key);
        }
        auto ms_reduce_ptr =
            reinterpret_cast<ReduceShaderT<Type>*>(&(*ms_reduce_it->second));

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

    template <NumericT KeyType,
              NumericT ValueType,
              typename ReduceOp = luisa::compute::Callable<Var<ValueType>(const Var<ValueType>&, const Var<ValueType>&)>>
    void compile_reduce_by_key_op(Device& device, ReduceOp reduce_op)
    {
        const size_t shared_mem_size = m_shared_mem_size;

        // auto segment_reduce_op = [&](SmemTypePtr<int>&       s_flags,
        //                              SmemTypePtr<ValueType>& s_reduced_values,
        //                              SmemTypePtr<int>&       s_segment_ids,
        //                              UInt                    num_item) noexcept
        // {
        //     Int thid = Int(thread_id().x);
        //     UInt block_start = block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);

        //     // block-level inclusive scan(ids)
        //     ArrayVar<int, ITEMS_PER_THREAD> local_flags;
        //     $for(i, 0u, UInt(ITEMS_PER_THREAD))
        //     {
        //         local_flags[i] = (*s_segment_ids)[thid * UInt(ITEMS_PER_THREAD) + i];
        //     };
        //     // BlockLoad<int, BLOCK_SIZE, ITEMS_PER_THREAD>().Load(s_segment_ids, local_flags);
        //     ArrayVar<int, ITEMS_PER_THREAD> output_segment_ids;
        //     BlockScan<int, BLOCK_SIZE, ITEMS_PER_THREAD>().ExclusiveSum(local_flags, output_segment_ids);
        //     $for(i, 0u, UInt(ITEMS_PER_THREAD))
        //     {
        //         UInt shared_idx = thid * UInt(ITEMS_PER_THREAD) + i;
        //         (*s_segment_ids)[shared_idx] = output_segment_ids[i];
        //     };
        //     sync_block();

        //     // segment reduce
        //     Local<uint> segment_starts{ITEMS_PER_THREAD};
        //     Local<uint> segment_ends{ITEMS_PER_THREAD};

        //     $for(i, 0u, UInt(ITEMS_PER_THREAD))
        //     {
        //         UInt shared_idx = thid * UInt(ITEMS_PER_THREAD) + i;
        //         UInt global_idx = block_start + shared_idx;

        //         $if(global_idx < num_item)
        //         {
        //             Int segment_id = (*s_segment_ids)[shared_idx];
        //             // find the start of the segment
        //             UInt start = shared_idx;
        //             UInt j     = UInt(shared_idx);
        //             $while(j > 0u)
        //             {
        //                 $if((*s_segment_ids)[j] == segment_id)
        //                 {
        //                     start = j;
        //                 }
        //                 $else
        //                 {
        //                     $break;
        //                 };
        //                 j = j - 1u;
        //             };
        //             segment_starts[i] = start;

        //             // find the end of the segment
        //             UInt end = shared_idx + 1u;
        //             j        = UInt(shared_idx + 1u);
        //             $while(j < block_size_x() * UInt(ITEMS_PER_THREAD))
        //             {
        //                 $if(j < num_item & (*s_segment_ids)[j] == segment_id)
        //                 {
        //                     end = j + 1u;
        //                 }
        //                 $else
        //                 {
        //                     $break;
        //                 };
        //                 j = j + 1u;
        //             };
        //             segment_ends[i] = end;
        //         };
        //     };
        //     sync_block();

        //     $for(step, 0u, 10u)
        //     {
        //         Int stride = 1 << step;

        //         $for(i, 0u, UInt(ITEMS_PER_THREAD))
        //         {
        //             UInt shared_idx = thid * UInt(ITEMS_PER_THREAD) + i;
        //             UInt global_idx = block_start + shared_idx;

        //             $if(global_idx < num_item)
        //             {
        //                 Int  segment_id = (*s_segment_ids)[shared_idx];
        //                 UInt start      = segment_starts[i];
        //                 UInt end        = segment_ends[i];

        //                 UInt rel_idx    = shared_idx - start;
        //                 Int  in_segment = rel_idx & (2 * stride - 1);
        //                 $if(in_segment == 0)
        //                 {
        //                     UInt parent_idx = shared_idx + stride;
        //                     $if(parent_idx < end & parent_idx < block_size_x() * UInt(ITEMS_PER_THREAD)
        //                         & (*s_segment_ids)[parent_idx] == segment_id)
        //                     {
        //                         // in the same segment
        //                         (*s_reduced_values)[shared_idx] +=
        //                             reduce_op((*s_reduced_values)[shared_idx],
        //                                       (*s_reduced_values)[parent_idx]);
        //                     };
        //                 };
        //             };
        //         };
        //         sync_block();
        //     };
        // };

        luisa::unique_ptr<ReduceByKeyShaderT<KeyType, ValueType>> ms_reduce_by_key_shader =
            nullptr;

        lazy_compile(
            device,
            ms_reduce_by_key_shader,
            [&](BufferVar<KeyType>   keys_in,
                BufferVar<ValueType> values_in,
                BufferVar<KeyType>   unique_out,
                BufferVar<ValueType> aggregated_out,
                BufferVar<uint>      num_runs_out,
                UInt                 num_item) noexcept
            {
                set_block_size(BLOCK_SIZE);
                UInt thid    = UInt(thread_id().x);
                UInt tile_id = block_id().x;
                UInt tile_offset = block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);
                SmemTypePtr<KeyType> s_keys = new SmemType<KeyType>{shared_mem_size};
                SmemTypePtr<ValueType> s_values = new SmemType<ValueType>{shared_mem_size};
                SmemTypePtr<int> s_discontinutity = new SmemType<int>{shared_mem_size};


                ArrayVar<KeyType, ITEMS_PER_THREAD>   local_keys;
                ArrayVar<ValueType, ITEMS_PER_THREAD> local_values;
                ArrayVar<int, ITEMS_PER_THREAD>       local_flags;

                // Bool is_last_tile = tile_id != block_size_x() - 1u;
                BlockLoad<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_keys).Load(
                    keys_in, local_keys, num_item);
                BlockLoad<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values).Load(
                    values_in, local_values, num_item);

                // load per keys
                Var<KeyType> tile_predecessor;
                $if(thid == 0)
                {
                    $if(block_id().x == 0)
                    {
                        tile_predecessor = local_keys[0];
                    }
                    $else
                    {
                        tile_predecessor = keys_in.read(tile_offset - 1);
                    };
                };
                sync_block();

                BlockDiscontinuity<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>().FlagHeads(
                    local_flags,
                    local_keys,
                    [&](const Var<KeyType>& a, const Var<KeyType>& b)
                    { return a != b; },
                    tile_predecessor);

                ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scan_items;
                $for(item, 0u, UInt(ITEMS_PER_THREAD))
                {
                    scan_items[item] = {local_flags[item], local_values[item]};
                };


                auto scan_by_key_op = [&](const Var<KeyValuePair<int, ValueType>>& a,
                                          const Var<KeyValuePair<int, ValueType>>& b)
                {
                    Var<KeyValuePair<int, ValueType>> result;
                    $if(b.key == 1)
                    {  // new segment
                        result.key   = a.key + 1;
                        result.value = b.value;
                    }
                    $else
                    {
                        result.key   = a.key;
                        result.value = reduce_op(a.value, b.value);
                    };
                    return result;
                };

                ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scan_output;
                Var<KeyValuePair<int, ValueType>> initial{0, 0};
                Var<KeyValuePair<int, ValueType>> block_aggregate{0, 0};

                BlockScan<KeyValuePair<int, ValueType>, BLOCK_SIZE, ITEMS_PER_THREAD>()
                    .ExclusiveScan(scan_items, scan_output, block_aggregate, scan_by_key_op, initial);

                // device_log("thid {},key {},value {},flags {}, scan_output_key {}, scan_output_value {}, block_aggregate key {}, value {}",
                //            block_id().x * block_size().x * UInt(ITEMS_PER_THREAD) + thid,
                //            local_keys[0],
                //            local_values[0],
                //            local_flags[0],
                //            scan_output[0].key,
                //            scan_output[0].value,
                //            block_aggregate.key,
                //            block_aggregate.value);
            });

        ms_reduce_by_key_map.try_emplace(get_key_value_op_shader_desc<KeyType, ValueType>(reduce_op),
                                         std::move(ms_reduce_by_key_shader));
    }

    template <NumericT KeyType,
              NumericT ValueType,
              typename ReduceOp = luisa::compute::Callable<Var<ValueType>(const Var<ValueType>&, const Var<ValueType>&)>>
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
            compile_reduce_by_key_op<KeyType, ValueType>(m_device, reduce_op);
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr =
            reinterpret_cast<ReduceByKeyShaderT<KeyType, ValueType>*>(
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