/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-17 22:53:38
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{
namespace details
{
    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    using namespace luisa::compute;
    template <typename Type4Byte, size_t BLOCK_SIZE = 256, size_t WARP_SIZE = 32>

    struct BlockReduceMem
    {
        template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        Var<Type4Byte> Reduce(SmemTypePtr<Type4Byte>& m_shared_mem,
                              const Var<Type4Byte>&   thread_data,
                              ReduceOp                reduce_op,
                              UInt                    valid_item = BLOCK_SIZE)
        {
            Var<Type4Byte> result;
            Int            thid   = Int(thread_id().x);
            (*m_shared_mem)[thid] = thread_data;
            sync_block();
            UInt stride = BLOCK_SIZE >> 1;
            $while(stride > 0)
            {
                $if(thid < stride)
                {
                    (*m_shared_mem)[thid] =
                        reduce_op((*m_shared_mem)[thid], (*m_shared_mem)[thid + stride]);
                };
                sync_block();
                stride >>= 1;
            };

            $if(thid == 0)
            {
                result = (*m_shared_mem)[0];
            };
            return result;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive