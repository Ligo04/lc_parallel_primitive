/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-23 00:06:30
 */

#pragma once

#include "luisa/core/logging.h"
#include "luisa/core/stl/memory.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/sugar.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/buffer.h"
#include <cstddef>
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/type_trait.h>
#include <limits>

namespace luisa::parallel_primitive
{
namespace detail
{

}
using namespace luisa::compute;

class DeviceReduce : public LuisaModule
{
    template <NumericT Type4Byte>
    using ReduceShaderT =
        Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int, int>;
    // Implementation details for DeviceReduce
  public:
    int m_block_size    = 256;
    int m_num_banks     = 32;
    int m_log_mem_banks = 5;

  private:
    int  m_shared_mem_size = 0;
    bool m_created         = false;

  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

    void create(Device& device)
    {
        int num_elements_per_block = m_block_size * 2;
        int extra_space            = num_elements_per_block / m_num_banks;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        compile<int>(device);
        compile<float>(device);
        m_created = true;
    }

    template <NumericT Type4Byte>
    void reduce(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, m_block_size, num_item);
        LUISA_ASSERT(temp_buffer.size() >= temp_storage_size,
                     "Please resize the temp buffer to at least {} elements, but got {} elements.",
                     temp_storage_size,
                     temp_buffer.size());
        reduce_array_recursive<Type4Byte>(
            cmdlist, temp_buffer, d_in, d_out, num_item, 0, 0, op);
    }

    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
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

  private:
    inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
    {
        return i >> m_log_mem_banks;
    }

    template <NumericT Type4Byte>
    void compile(Device& device)
    {

        const auto n_blocks        = m_block_size;
        size_t     shared_mem_size = m_shared_mem_size;
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
            Int ai     = thid;
            Int bi     = ai + block_dim_x;  // bank conflict free

            Int bank_offset_a = conflict_free_offset(ai);
            Int bank_offset_b = conflict_free_offset(bi);

            Var<Type4Byte> initial;
            $if(op == 0)
            {
                initial = Type4Byte(0);
            }
            $elif(op == 1)
            {
                initial = std::numeric_limits<Type4Byte>::min();
            }
            $elif(op == 2)
            {
                initial = std::numeric_limits<Type4Byte>::max();
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
        auto up_sweep_op = [&](SmemTypePtr<Type4Byte>& s_data, Int n, Int op)
        {
            Int thid   = Int(thread_id().x);
            Int stride = def(1);

            Int d = Int(block_size_x());
            $if(d > 0)
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
                        (*s_data)[bi] = max((*s_data)[bi], (*s_data)[ai]);
                    }
                    $elif(op == 2)
                    {
                        (*s_data)[bi] = min((*s_data)[bi], (*s_data)[ai]);
                    };
                };
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
                    g_blockSums.write(blockIndex, (*s_data)[index]);
                };
                (*s_data)[index] = Type4Byte(0);  // zero the last element in the scan so it will propagate back to the front
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
            up_sweep_op(s_data, n, op);
            clear_last_element(1, s_data, block_sums, block_index);
        };

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


    // for reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
};
}  // namespace luisa::parallel_primitive