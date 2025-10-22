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

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <typename Type4Byte, size_t WARP_SIZE = 32>
    struct WarpReduceShfl
    {
        template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        Var<Type4Byte> Reduce(const Var<Type4Byte>& d_in, ReduceOp op, UInt valid_item = WARP_SIZE)
        {
            Var<Type4Byte> result    = d_in;
            compute::UInt  lane_id   = compute::warp_lane_id();
            compute::UInt  wave_size = compute::warp_lane_count();

            compute::UInt offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = compute::warp_read_lane(result, lane_id + offset);
                $if(lane_id + offset < valid_item)
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