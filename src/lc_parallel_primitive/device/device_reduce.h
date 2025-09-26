/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-26 18:10:01
 */

#pragma once
#include "luisa/ast/type.h"
#include "luisa/runtime/stream.h"
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
#include <limits>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
class DeviceReduce : public LuisaModule
{
    template <NumericT Type4Byte>
    using KVTP = KeyValuePair<int, Type4Byte>;

    template <NumericT Type4Byte>
    using ReduceShaderT =
        Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int, int>;
    template <NumericT Type4Byte>
    using ArgReduceShaderT =
        Shader<1, Buffer<KVTP<Type4Byte>>, Buffer<KVTP<Type4Byte>>, int, int, int, int>;

    template <NumericT Type4Byte>
    using ConstructKVPShaderT = Shader<1, Buffer<Type4Byte>, Buffer<KVTP<Type4Byte>>>;

    template <NumericT Type4Byte>
    using AssignKVPShaderT =
        Shader<1, Buffer<KVTP<Type4Byte>>, Buffer<Type4Byte>, Buffer<int>>;

  public:
    int m_block_size    = 256;
    int m_num_banks     = 32;
    int m_log_mem_banks = 5;

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
        int extra_space            = num_elements_per_block / m_num_banks;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        compile<int>(device);
        compile<float>(device);
#ifdef __WIN32
        compile<double>(device);
#endif
        m_created = true;
    }

    template <NumericT Type4Byte>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        Buffer<Type4Byte> temp_buffer = m_device.create_buffer<Type4Byte>(temp_storage_size);
        reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in, d_out, num_item, 0, 0, op);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, stream, d_in, d_out, num_item, 0);
    }

    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, stream, d_in, d_out, num_item, 1);
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, stream, d_in, d_out, num_item, 2);
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<int>       d_index_out,
                size_t                num_item)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        // key value pair reduce
        Buffer<KVTP<Type4Byte>> d_in_kv =
            m_device.create_buffer<KVTP<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_kvp_construct_it = ms_kvp_construct_map.find(key);
        auto ms_kvp_construct_ptr = reinterpret_cast<ConstructKVPShaderT<Type4Byte>*>(
            &(*ms_kvp_construct_it->second));
        cmdlist << (*ms_kvp_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());

        Buffer<KVTP<Type4Byte>> temp_buffer =
            m_device.create_buffer<KVTP<Type4Byte>>(temp_storage_size);
        Buffer<KVTP<Type4Byte>> d_out_kv = m_device.create_buffer<KVTP<Type4Byte>>(1);
        arg_reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in_kv.view(), d_out_kv.view(), num_item, 0, 0, 0);

        // copy result to d_out and d_index_out
        auto ms_kvp_assign_it = ms_kvp_assign_map.find(key);
        auto ms_kvp_assign_ptr =
            reinterpret_cast<AssignKVPShaderT<Type4Byte>*>(&(*ms_kvp_assign_it->second));
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
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        // key value pair reduce
        Buffer<KVTP<Type4Byte>> d_in_kv =
            m_device.create_buffer<KVTP<Type4Byte>>(d_in.size());

        // construct key value pair
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_kvp_construct_it = ms_kvp_construct_map.find(key);
        auto ms_kvp_construct_ptr = reinterpret_cast<ConstructKVPShaderT<Type4Byte>*>(
            &(*ms_kvp_construct_it->second));
        cmdlist << (*ms_kvp_construct_ptr)(d_in, d_in_kv.view()).dispatch(d_in.size());

        Buffer<KVTP<Type4Byte>> temp_buffer =
            m_device.create_buffer<KVTP<Type4Byte>>(temp_storage_size);
        Buffer<KVTP<Type4Byte>> d_out_kv = m_device.create_buffer<KVTP<Type4Byte>>(1);
        arg_reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer.view(), d_in_kv.view(), d_out_kv.view(), num_item, 0, 0, 1);

        // copy result to d_out and d_index_out
        auto ms_kvp_assign_it = ms_kvp_assign_map.find(key);
        auto ms_kvp_assign_ptr =
            reinterpret_cast<AssignKVPShaderT<Type4Byte>*>(&(*ms_kvp_assign_it->second));
        cmdlist << (*ms_kvp_assign_ptr)(d_out_kv.view(), d_out, d_index_out)
                       .dispatch(d_index_out.size());

        // Stream stream = m_device.create_stream();
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte>
    void ReduceByKey(CommandList&          cmdlist,
                     BufferView<Type4Byte> temp_buffer,
                     BufferView<Type4Byte> d_keys_in,
                     BufferView<Type4Byte> d_values_in,
                     BufferView<Type4Byte> d_keys_out,
                     BufferView<Type4Byte> d_values_out,
                     size_t                num_item,
                     int                   op = 0)
    {
    }


    template <NumericT Type4Byte>
    void TransformReduce(CommandList&          cmdlist,
                         BufferView<Type4Byte> temp_buffer,
                         BufferView<Type4Byte> d_keys_in,
                         BufferView<Type4Byte> d_values_in,
                         BufferView<Type4Byte> d_keys_out,
                         BufferView<Type4Byte> d_values_out,
                         size_t                num_item,
                         int                   op = 0)
    {
    }

  private:
    inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
    {
        return i >> m_log_mem_banks;
    }

    // Callable create_index_key_value_pair = []()

    template <NumericT Type4Byte>
    void compile(Device& device)
    {

        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;
        // for construct key value pair
        luisa::unique_ptr<ConstructKVPShaderT<Type4Byte>> ms_construct_key_value_pair_shader =
            nullptr;
        lazy_compile(device,
                     ms_construct_key_value_pair_shader,
                     [&](BufferVar<Type4Byte> arr_in, BufferVar<KVTP<Type4Byte>> g_kv_out) noexcept
                     {
                         set_block_size(n_blocks);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);

                         Var<KVTP<Type4Byte>> initial{0, 0};
                         initial.index = global_id;
                         initial.value = arr_in.read(global_id);
                         g_kv_out.write(global_id, initial);
                     });
        ms_kvp_construct_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_construct_key_value_pair_shader));


        luisa::unique_ptr<AssignKVPShaderT<Type4Byte>> ms_kvp_assign_shader = nullptr;
        lazy_compile(device,
                     ms_kvp_assign_shader,
                     [&](BufferVar<KVTP<Type4Byte>> kvp_in,
                         BufferVar<Type4Byte>       value_out,
                         BufferVar<int>             index_out) noexcept
                     {
                         set_block_size(n_blocks);
                         Int global_id =
                             Int(block_id().x * block_size().x + thread_id().x);

                         Var<KVTP<Type4Byte>> kvp = kvp_in.read(global_id);
                         index_out.write(global_id, kvp.index);
                         value_out.write(global_id, kvp.value);
                     });
        ms_kvp_assign_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_kvp_assign_shader));

        // for reduce
        auto load_shared_chunk_from_mem_op = [&](SmemTypePtr<Type4Byte>& s_data,
                                                 BufferVar<Type4Byte>& g_idata,
                                                 Int                   n,
                                                 Int& baseIndex,
                                                 Int  op)
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

            Var<Type4Byte> initial;
            $if(op == 0)
            {
                initial = Type4Byte(0);
            }
            $elif(op == 1)
            {
                initial = std::numeric_limits<Type4Byte>::max();
            }
            $elif(op == 2)
            {
                initial = std::numeric_limits<Type4Byte>::min();
            };

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

        auto reduce_op = [&](SmemTypePtr<Type4Byte>& s_data, Int n, Int op)
        {
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
                    $if(op == 0)
                    {
                        (*s_data)[bi] += (*s_data)[ai];
                    }
                    $elif(op == 1)
                    {
                        (*s_data)[bi] = min((*s_data)[bi], (*s_data)[ai]);
                    }
                    $elif(op == 2)
                    {
                        (*s_data)[bi] = max((*s_data)[bi], (*s_data)[ai]);
                    };
                };
                stride *= 2;
                d = d >> 1;
            };
        };

        auto clear_last_element = [&](Int                     storeSum,
                                      SmemTypePtr<Type4Byte>& s_data,
                                      BufferVar<Type4Byte>&   g_blockSums,
                                      Int                     blockIndex)
        {
            Int thid = Int(thread_id().x);
            Int d    = Int(block_size_x());
            $if(thid == 0)
            {
                Int index = (d << 1) - 1;
                index += conflict_free_offset(index);
                $if(storeSum == 1)
                {
                    // write this block's total sum to the corresponding index in the blockSums array
                    g_blockSums.write(blockIndex, (*s_data)[index]);
                };
                // zero the last element in the scan so it will propagate back to the front
                (*s_data)[index] = Type4Byte(0);
            };
        };

        auto reduce_block = [&](SmemTypePtr<Type4Byte>& s_data,
                                BufferVar<Type4Byte>&   block_sums,
                                Int                     block_index,
                                Int                     n,
                                Int                     op)
        {
            $if(block_index == 0)
            {
                block_index = Int(block_id().x);
            };
            // build the op in place up the tree
            reduce_op(s_data, n, op);
            clear_last_element(1, s_data, block_sums, block_index);
        };
        //  reduce
        luisa::unique_ptr<ReduceShaderT<Type4Byte>> ms_reduce = nullptr;
        lazy_compile(device,
                     ms_reduce,
                     [&](BufferVar<Type4Byte> g_idata,
                         BufferVar<Type4Byte> g_block_sums,
                         Int                  n,
                         Int                  block_index,
                         Int                  base_index,
                         Int                  op) noexcept
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
                         load_shared_chunk_from_mem_op(s_data, g_idata, n, base_index, op);
                         reduce_block(s_data, g_block_sums, block_index, n, op);
                     });

        ms_reduce_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_reduce));


        auto load_shared_chunk_from_mem_op_arg = [&](SmemTypePtr<KVTP<Type4Byte>>& s_data,
                                                     BufferVar<KVTP<Type4Byte>>& g_idata,
                                                     Int  n,
                                                     Int& baseIndex,
                                                     Int  op)
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

            Var<KVTP<Type4Byte>> initial{0, 0};
            $if(op == 0)
            {
                initial.index = thread_id_x;
                initial.value = std::numeric_limits<Type4Byte>::max();
            }
            $elif(op == 1)
            {
                initial.index = thread_id_x;
                initial.value = std::numeric_limits<Type4Byte>::min();
            };

            Var<KVTP<Type4Byte>> data_ai = initial;
            Var<KVTP<Type4Byte>> data_bi = initial;

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

        // arg reduce
        auto min_arg_op = [&](Var<KVTP<Type4Byte>>& a, Var<KVTP<Type4Byte>>& b) noexcept
        {
            $if(b.value < a.value | (b.value == a.value & b.index < a.index))
            {
                a = b;
            };
            return a;
        };

        auto max_arg_op = [&](Var<KVTP<Type4Byte>>& a, Var<KVTP<Type4Byte>>& b) noexcept
        {
            $if(b.value > a.value | (b.value == a.value & b.index < a.index))
            {
                a = b;
            };
            return a;
        };

        auto arg_reduce_op = [&](SmemTypePtr<KVTP<Type4Byte>>& s_data, Int n, Int op)
        {
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
                    $if(op == 0)
                    {
                        (*s_data)[bi] = min_arg_op((*s_data)[bi], (*s_data)[ai]);
                    }
                    $elif(op == 1)
                    {
                        (*s_data)[bi] = max_arg_op((*s_data)[bi], (*s_data)[ai]);
                    };
                };
                stride *= 2;
                d = d >> 1;
            };
        };

        auto clear_last_element_arg = [&](Int storeSum,
                                          SmemTypePtr<KVTP<Type4Byte>>& s_data,
                                          BufferVar<KVTP<Type4Byte>>& g_blockSums,
                                          Int blockIndex)
        {
            Int thid = Int(thread_id().x);
            Int d    = Int(block_size_x());
            $if(thid == 0)
            {
                Int index = (d << 1) - 1;
                index += conflict_free_offset(index);
                $if(storeSum == 1)
                {
                    // write this block's total sum to the corresponding index in the blockSums array
                    g_blockSums.write(blockIndex, (*s_data)[index]);
                };
                // zero the last element in the scan so it will propagate back to the front
                (*s_data)[index].value = Type4Byte(0);
                (*s_data)[index].index = 0;
            };
        };

        auto arg_reduce_block = [&](SmemTypePtr<KVTP<Type4Byte>>& s_data,
                                    BufferVar<KVTP<Type4Byte>>&   block_sums,
                                    Int                           block_index,
                                    Int                           n,
                                    Int                           op)
        {
            $if(block_index == 0)
            {
                block_index = Int(block_id().x);
            };
            // build the op in place up the tree
            arg_reduce_op(s_data, n, op);
            clear_last_element_arg(1, s_data, block_sums, block_index);
        };
        luisa::unique_ptr<ArgReduceShaderT<Type4Byte>> ms_arg_reduce = nullptr;
        lazy_compile(device,
                     ms_arg_reduce,
                     [&](BufferVar<KVTP<Type4Byte>> g_idata,
                         BufferVar<KVTP<Type4Byte>> g_block_sums,
                         Int                        n,
                         Int                        block_index,
                         Int                        base_index,
                         Int                        op) noexcept
                     {
                         set_block_size(n_blocks);
                         Int ai, bi, men_ai, men_bi, bank_offset_a, bank_offset_b;
                         Int block_id_x  = Int(block_id().x);
                         Int block_dim_x = Int(block_size_x());

                         SmemTypePtr<KVTP<Type4Byte>> s_data =
                             new SmemType<KVTP<Type4Byte>>{shared_mem_size};

                         $if(base_index == 0)
                         {
                             base_index = block_id_x * (block_dim_x << 1);
                         };
                         load_shared_chunk_from_mem_op_arg(s_data, g_idata, n, base_index, op);
                         arg_reduce_block(s_data, g_block_sums, block_index, n, op);
                     });
        ms_arg_reduce_map.try_emplace(
            luisa::string{luisa::compute::Type::of<Type4Byte>()->description()},
            std::move(ms_arg_reduce));
        // reduce by key
    }


    template <NumericT Type4Byte>
    void reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                BufferView<Type4Byte>        temp_storage,
                                BufferView<Type4Byte>        arr_in,
                                BufferView<Type4Byte>        arr_out,
                                int                          num_elements,
                                int                          offset,
                                int                          level,
                                int                          op) noexcept
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
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_reduce_it = ms_reduce_map.find(key);
        auto ms_reduce_ptr =
            reinterpret_cast<ReduceShaderT<Type4Byte>*>(&(*ms_reduce_it->second));

        if(num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, op)
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(arr_in,
                                            temp_buffer_level,
                                            num_elements_last_block,
                                            num_blocks - 1,
                                            num_elements - num_elements_last_block,
                                            op)
                               .dispatch(block_size);
            }

            reduce_array_recursive<Type4Byte>(cmdlist,
                                              temp_buffer_level,
                                              temp_buffer_level,
                                              arr_out,
                                              num_blocks,
                                              num_blocks,
                                              level + 1,
                                              op);
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, op)
                           .dispatch(block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };


    template <NumericT Type4Byte>
    void arg_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                    BufferView<KVTP<Type4Byte>>  temp_storage,
                                    BufferView<KVTP<Type4Byte>>  arr_in,
                                    BufferView<KVTP<Type4Byte>>  arr_out,
                                    int                          num_elements,
                                    int                          offset,
                                    int                          level,
                                    int                          op) noexcept
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
        BufferView<KVTP<Type4Byte>> temp_buffer_level =
            temp_storage.subview(offset, size_elements);
        // execute the scan
        auto key = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_arg_reduce_it = ms_arg_reduce_map.find(key);
        auto ms_arg_reduce_ptr =
            reinterpret_cast<ArgReduceShaderT<Type4Byte>*>(&(*ms_arg_reduce_it->second));

        if(num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_arg_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, op)
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if(np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_arg_reduce_ptr)(arr_in,
                                                temp_buffer_level,
                                                num_elements_last_block,
                                                num_blocks - 1,
                                                num_elements - num_elements_last_block,
                                                op)
                               .dispatch(block_size);
            }

            arg_reduce_array_recursive<Type4Byte>(cmdlist,
                                                  temp_buffer_level,
                                                  temp_buffer_level,
                                                  arr_out,
                                                  num_blocks,
                                                  num_blocks,
                                                  level + 1,
                                                  op);
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_arg_reduce_ptr)(arr_in, temp_buffer_level, num_elements, 0, 0, op)
                           .dispatch(block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    };


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;

    // for key value pair
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_kvp_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_kvp_assign_map;
};
}  // namespace luisa::parallel_primitive