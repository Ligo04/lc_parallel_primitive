/*
 * @Author: Ligo 
 * @Date: 2025-10-17 16:22:56 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:57:01
 */


#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <typename Type4Byte, size_t LOGIC_WARP_SIZE = details::WARP_SIZE>
    struct WarpReduceShfl
    {
        constexpr static bool IS_ARCH_WARP = (LOGIC_WARP_SIZE == details::WARP_SIZE);
        compute::UInt         lane_id;
        compute::UInt         warp_id;
        compute::UInt         member_mask;

        WarpReduceShfl()
            : lane_id(warp_lane_id())
            , warp_id(IS_ARCH_WARP ? 0 : warp_lane_id() / compute::UInt(LOGIC_WARP_SIZE))
            , member_mask(IS_ARCH_WARP ? 0xFFFFFFFFu : warp_mask<LOGIC_WARP_SIZE>(warp_id))
        {
            if constexpr(!IS_ARCH_WARP)
            {
                lane_id = lane_id % compute::UInt(LOGIC_WARP_SIZE);
            }
        }

        template <typename ReduceOp>
        Var<Type4Byte> Reduce(const Var<Type4Byte>& input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE)
        {
            return ReduceStep(input, op, valid_item - 1);
        }

        template <bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
        Var<Type4Byte> SegmentReduce(const Var<Type4Byte>& input,
                                     const Var<FlagT>&     flag,
                                     ReduceOp              reduce_op,
                                     UInt                  valid_item = LOGIC_WARP_SIZE)
        {
            UInt warp_flags = compute::warp_active_bit_mask(flag == 1u).x;
            if constexpr(HEAD_SEGMENT)
            {
                warp_flags >>= 1;
            };

            warp_flags &= get_lane_mask_ge();

            if constexpr(!IS_ARCH_WARP)
            {
                warp_flags = (warp_flags & member_mask) >> (warp_id * UInt(LOGIC_WARP_SIZE));
            };

            warp_flags |= 1u << (UInt(LOGIC_WARP_SIZE) - 1u);

            UInt last_lane = compute::ctz(warp_flags);

            return ReduceStep(input, reduce_op, last_lane);
        }

      private:
        template <typename ReduceOp>
        Var<Type4Byte> ReduceStep(const Var<Type4Byte>& input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE)
        {
            Var<Type4Byte> result = input;

            compute::UInt offset = 1u;
            $while(offset < compute::warp_lane_count())
            {
                Var<Type4Byte> temp = ShuffleDown(result, lane_id, offset, valid_item);
                $if(lane_id + offset <= valid_item)
                {
                    result = op(result, temp);
                };
                offset <<= 1;
            };
            return result;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive