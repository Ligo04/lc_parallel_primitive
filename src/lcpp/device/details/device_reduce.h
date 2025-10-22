/*
 * @Author: Ligo 
 * @Date: 2025-10-21 23:03:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:55:23
 */

#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename T>
    concept NumericTOrKeyValuePairT = NumericT<T> || KeyValuePairType<T>;
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

    template <NumericTOrKeyValuePairT DataType, size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
    class ReduceShader : public LuisaModule
    {
      public:
        template <typename ReduceOp>
        luisa::unique_ptr<ReduceShaderT<DataType>> compile(Device& device,
                                                           size_t shared_mem_size,
                                                           ReduceOp reduce_op)
        {
            // for reduce
            auto load_shared_chunk_from_mem = [&](SmemTypePtr<DataType>& s_data,
                                                  BufferVar<DataType>& g_idata,
                                                  Int                  n,
                                                  Int           baseIndex,
                                                  Var<DataType> initial)
            {
                Int thread_id_x = Int(thread_id().x);
                Int block_id_x  = Int(block_id().x);
                Int block_dim_x = Int(block_size_x());

                // $for(item, 0u, UInt(ITEMS_PER_THREAD))
                for(auto item = 0u; item < ITEMS_PER_THREAD; ++item)
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

            auto reduce_block = [&](SmemTypePtr<DataType>& s_data,
                                    BufferVar<DataType>&   block_sums,
                                    Int                    block_index)
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
            lazy_compile(device,
                         ms_reduce_shader,
                         [&](BufferVar<DataType> g_idata,
                             BufferVar<DataType> g_block_sums,
                             Int                 num_elements,
                             Int                 block_index,
                             Int                 base_index,
                             Var<DataType>       init) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             Int block_id_x  = Int(block_id().x);
                             Int block_dim_x = Int(block_size_x());
                             SmemTypePtr<DataType> s_data =
                                 new SmemType<DataType>{shared_mem_size};

                             $if(base_index == 0)
                             {
                                 base_index = block_id_x * (block_dim_x * UInt(ITEMS_PER_THREAD));
                             };
                             load_shared_chunk_from_mem(
                                 s_data, g_idata, num_elements, base_index, init);
                             reduce_block(s_data, g_block_sums, block_index);
                         });

            return ms_reduce_shader;
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive