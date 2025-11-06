/*
 * @Author: Ligo 
 * @Date: 2025-10-21 23:03:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 14:31:03
 */

#pragma once
#include "luisa/dsl/stmt.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/common/utils.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE>
    class ArgReduce : public LuisaModule
    {
      public:
        using ArgConstructShaderT = Shader<1, Buffer<Type4Byte>, Buffer<IndexValuePairT<Type4Byte>>>;

        using ArgAssignShaderT =
            Shader<1, Buffer<IndexValuePairT<Type4Byte>>, Buffer<Type4Byte>, Buffer<uint>>;

        U<ArgConstructShaderT> compile_arg_construct_shader(Device& device)
        {
            U<ArgConstructShaderT> ms_arg_construct_shader = nullptr;
            lazy_compile(device,
                         ms_arg_construct_shader,
                         [&](BufferVar<Type4Byte> arr_in, BufferVar<IndexValuePairT<Type4Byte>> g_kv_out)
                         {
                             set_block_size(BLOCK_SIZE);
                             Int global_id = Int(block_id().x * block_size().x + thread_id().x);

                             Var<IndexValuePairT<Type4Byte>> initial{0, 0};
                             initial.key   = global_id;
                             initial.value = arr_in.read(global_id);
                             g_kv_out.write(global_id, initial);
                         });
            return ms_arg_construct_shader;
        }

        U<ArgAssignShaderT> compile_arg_assign_shader(Device& device)
        {
            U<ArgAssignShaderT> ms_arg_assign_shader = nullptr;
            lazy_compile(device,
                         ms_arg_assign_shader,
                         [&](BufferVar<IndexValuePairT<Type4Byte>> kvp_in,
                             BufferVar<Type4Byte>                  value_out,
                             BufferVar<uint>                       index_out)
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt global_id = UInt(block_id().x * block_size().x + thread_id().x);

                             Var<IndexValuePairT<Type4Byte>> kvp = kvp_in.read(global_id);
                             index_out.write(global_id, kvp.key);
                             value_out.write(global_id, kvp.value);
                         });
            return ms_arg_assign_shader;
        }
    };


    template <NumericTOrKeyValuePairT DataType, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class ReduceShader : public LuisaModule
    {
      public:
        using ReduceShaderKernel = Shader<1, Buffer<DataType>, Buffer<DataType>, uint, uint, uint, DataType>;

        template <typename ReduceOp, typename TransformOp>
        U<ReduceShaderKernel> compile(Device& device, size_t shared_mem_size, ReduceOp reduce_op, TransformOp transform_op)
        {
            // for reduce
            auto load_shared_chunk_from_mem = [&](SmemTypePtr<DataType>& s_data,
                                                  BufferVar<DataType>&   g_idata,
                                                  UInt                   n,
                                                  UInt                   baseIndex,
                                                  Var<DataType>          initial_value)
            {
                UInt thread_id_x = thread_id().x;
                UInt block_id_x  = block_id().x;
                UInt block_dim_x = block_size_x();

                for(auto item = 0u; item < ITEMS_PER_THREAD; ++item)
                {
                    UInt          shared_idx = thread_id_x + item * block_dim_x;
                    UInt          global_idx = baseIndex + shared_idx;
                    Var<DataType> data       = initial_value;
                    $if(global_idx < n)
                    {
                        data = g_idata.read(global_idx);
                    };

                    Int bank_offset                     = conflict_free_offset(shared_idx);
                    (*s_data)[shared_idx + bank_offset] = transform_op(data);
                };
            };

            auto reduce_block = [&](SmemTypePtr<DataType>& s_data, BufferVar<DataType>& block_sums, UInt block_index)
            {
                $if(block_index == 0)
                {
                    block_index = block_id().x;
                };
                // build the op in place up the tree
                UInt thid   = thread_id().x;
                UInt stride = def(1);
                // build the sum in place up the tree
                UInt d = block_size_x();
                $while(d > 0)
                {
                    sync_block();
                    $if(thid < d)
                    {
                        UInt i  = (stride * 2) * thid;
                        UInt ai = i + stride - 1;
                        UInt bi = ai + stride;

                        ai += conflict_free_offset(ai);
                        bi += conflict_free_offset(bi);
                        $if(bi < UInt(shared_mem_size) & ai < UInt(shared_mem_size))
                        {
                            (*s_data)[bi] = reduce_op((*s_data)[ai], (*s_data)[bi]);
                        };
                    };
                    stride *= 2;
                    d = d >> 1;
                };

                $if(thid == 0)
                {
                    UInt index = block_size_x() * UInt(ITEMS_PER_THREAD) - 1;
                    index += conflict_free_offset(index);
                    block_sums.write(block_index, (*s_data)[index]);
                };
            };

            //  reduce
            U<ReduceShaderKernel> ms_reduce_shader = nullptr;
            lazy_compile(device,
                         ms_reduce_shader,
                         [&](BufferVar<DataType> g_idata,
                             BufferVar<DataType> g_block_sums,
                             UInt                num_elements,
                             UInt                block_index,
                             UInt                base_index,
                             Var<DataType>       initial_value) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt                  block_id_x  = block_id().x;
                             UInt                  block_dim_x = block_size_x();
                             SmemTypePtr<DataType> s_data = new SmemType<DataType>{shared_mem_size};

                             $if(base_index == 0)
                             {
                                 base_index = block_id_x * (block_dim_x * UInt(ITEMS_PER_THREAD));
                             };
                             load_shared_chunk_from_mem(s_data, g_idata, num_elements, base_index, initial_value);
                             reduce_block(s_data, g_block_sums, block_index);
                         });

            return ms_reduce_shader;
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive