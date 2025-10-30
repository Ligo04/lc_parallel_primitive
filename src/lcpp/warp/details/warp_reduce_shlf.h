/*
 * @Author: Ligo 
 * @Date: 2025-10-17 16:22:56 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:57:01
 */


#pragma once
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/stmt.h"
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

        template <typename ReduceOp>
        Var<Type4Byte> Reduce(const Var<Type4Byte>& input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE)
        {
            Var<Type4Byte> result    = input;
            compute::UInt  lane_id   = compute::warp_lane_id();
            compute::UInt  wave_size = compute::warp_lane_count();

            compute::UInt offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = ShuffleDown(result, lane_id, offset, valid_item);
                $if(lane_id + offset < valid_item)
                {
                    result = op(result, temp);
                };
                offset <<= 1;
            };
            return result;
        }


        template <bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
        Var<Type4Byte> SegmentReduce(const Var<Type4Byte>& input,
                                     const Var<FlagT>&     flag,
                                     ReduceOp              redecu_op,
                                     UInt valid_item = LOGIC_WARP_SIZE)
        {
            compute::UInt lane_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            UInt warp_flags = compute::warp_active_bit_mask(flag == 1).x;
            if constexpr(HEAD_SEGMENT)
            {
                warp_flags >>= 1;
            };

            if constexpr(!IS_ARCH_WARP)
            {
                compute::UInt member_mask = warp_mask<LOGIC_WARP_SIZE>(lane_id);
                compute::UInt warp_id = lane_id / compute::UInt(LOGIC_WARP_SIZE);
                warp_flags = (warp_flags & member_mask)
                             >> (warp_id * UInt(LOGIC_WARP_SIZE));
            };

            warp_flags &= get_lane_mask_ge(lane_id, wave_size);

            warp_flags |= 1u << (wave_size - 1u);

            UInt last_lane = compute::clz(compute::reverse(warp_flags));

            return Reduce(input, redecu_op, last_lane + 1);
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive