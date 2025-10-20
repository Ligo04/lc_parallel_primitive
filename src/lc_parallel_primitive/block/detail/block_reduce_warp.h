/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-17 18:45:35
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

    struct BlockReduceShfl
    {
        template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        Var<Type4Byte> Reduce(SmemTypePtr<Type4Byte>& m_shared_mem,
                              const Var<Type4Byte>&   thread_data,
                              ReduceOp                reduce_op,
                              UInt                    valid_item = BLOCK_SIZE)
        {
            UInt warp_id     = thread_id().x / warp_lane_count();
            UInt warp_offset = warp_id * warp_lane_count();
            UInt warp_nums   = ceil(BLOCK_SIZE / float(WARP_SIZE));
            UInt lane_id     = warp_lane_id();
            UInt thid        = thread_id().x;

            UInt warp_valid_num = warp_lane_count();
            $if(warp_offset + warp_lane_count() > valid_item)
            {
                warp_valid_num = valid_item - warp_offset;
            };

            Var<Type4Byte> warp_aggregate =
                WarpReduce<Type4Byte, WARP_SIZE>().Reduce(thread_data, reduce_op, warp_valid_num);

            // write warp result to shared memory
            $if(lane_id == 0)
            {
                (*m_shared_mem)[warp_id] = warp_aggregate;
            };

            sync_block();

            $if(thid == 0)
            {
                $for(warp_idx, 1u, warp_nums)
                {
                    $if(warp_idx * warp_lane_count() < valid_item)
                    {
                        Var<Type4Byte> val = (*m_shared_mem)[warp_idx];
                        warp_aggregate     = reduce_op(warp_aggregate, val);
                    };
                };
            };

            return warp_aggregate;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive