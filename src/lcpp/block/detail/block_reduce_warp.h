/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 17:04:22
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    template <typename Type4Byte, size_t BLOCK_SIZE = 256, size_t WARP_SIZE = 32>
    struct BlockReduceShfl
    {
        template <bool IS_FULL_TILE,
                  typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        Var<Type4Byte> Reduce(SmemTypePtr<Type4Byte>& m_shared_mem,
                              const Var<Type4Byte>&   thread_data,
                              ReduceOp                reduce_op,
                              UInt                    valid_item)
        {
            UInt warp_id     = thread_id().x / warp_lane_count();
            UInt warp_offset = warp_id * warp_lane_count();
            uint warp_nums   = ceil(BLOCK_SIZE / WARP_SIZE);
            UInt lane_id     = warp_lane_id();
            UInt thid        = thread_id().x;

            UInt warp_valid_num = warp_lane_count();
            $if(!IS_FULL_TILE & warp_offset + warp_lane_count() > valid_item)
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
                // $for(warp_idx, 1u, warp_nums)
                for(int warp_idx = 1u; warp_idx < warp_nums; ++warp_idx)
                {
                    $if(IS_FULL_TILE | warp_idx * warp_lane_count() < valid_item)
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