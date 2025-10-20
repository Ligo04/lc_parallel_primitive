/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:00:46 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 14:56:00
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
    struct WarpScanShfl
    {
        template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            compute::UInt lane_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            Var<Type4Byte> output = thread_input;
            compute::UInt  offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = compute::warp_read_lane(output, lane_id - offset);
                $if(lane_id >= offset)
                {
                    output = scan_op(temp, output);
                };
                offset <<= 1;
            };
            inclusive_output = output;
        }
        template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op)
        {
            InclusiveScan(thread_input, inclusive_output, scan_op, Type4Byte(0));
        }

        template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value = Type4Byte(0))
        {
            Var<Type4Byte> inclusive_output;
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output =
                compute::warp_read_lane(inclusive_output, compute::warp_lane_id() - 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
        }
        template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value = Type4Byte(0))
        {
            Var<Type4Byte> inclusive_output;
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output =
                compute::warp_read_lane(inclusive_output, compute::warp_lane_id() - 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
            warp_aggregate =
                compute::warp_read_lane(inclusive_output, warp_lane_id() - 1u);
        }


        template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
        void Scan(const Var<Type4Byte>& thread_input,
                  Var<Type4Byte>&       inclusive_output,
                  Var<Type4Byte>&       exclusive_output,
                  ScanOp                scan_op,
                  const Var<Type4Byte>& initial_value = Type4Byte(0))
        {
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output =
                compute::warp_read_lane(inclusive_output, compute::warp_lane_id() - 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive