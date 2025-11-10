/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:00:46 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:57:08
 */


#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename Type4Byte, size_t LOGIC_WARP_SIZE = details::WARP_SIZE>
    struct WarpScanShfl
    {
        template <typename ScanOp>
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
                Var<Type4Byte> temp = ShuffleUp(output, lane_id, offset);

                $if(lane_id >= offset)
                {
                    output = scan_op(temp, output);
                };
                offset <<= 1;
            };
            inclusive_output = scan_op(initial_value, output);
        }


        template <typename ScanOp>
        void InclusiveScan(const Var<Type4Byte>& thread_input, Var<Type4Byte>& inclusive_output, ScanOp scan_op)
        {
            compute::UInt lane_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            Var<Type4Byte> output = thread_input;
            compute::UInt  offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = ShuffleUp(output, lane_id, offset, 0u);
                $if(lane_id >= offset)
                {
                    output = scan_op(temp, output);
                };
                offset <<= 1;
            };
            inclusive_output = output;
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> inclusive_output;
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output = ShuffleUp(inclusive_output, compute::warp_lane_id(), 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> inclusive_output;
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output = ShuffleUp(inclusive_output, compute::warp_lane_id(), 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
            warp_aggregate = ShuffleUp(inclusive_output, compute::warp_lane_count(), 1u);
        }

        template <typename ScanOp>
        void Scan(const Var<Type4Byte>& thread_input,
                  Var<Type4Byte>&       inclusive_output,
                  Var<Type4Byte>&       exclusive_output,
                  ScanOp                scan_op)
        {
            InclusiveScan(thread_input, inclusive_output, scan_op);
            exclusive_output = ShuffleUp(inclusive_output, compute::warp_lane_id(), 1u);
        }

        template <typename ScanOp>
        void Scan(const Var<Type4Byte>& thread_input,
                  Var<Type4Byte>&       inclusive_output,
                  Var<Type4Byte>&       exclusive_output,
                  ScanOp                scan_op,
                  const Var<Type4Byte>& initial_value)
        {
            InclusiveScan(thread_input, inclusive_output, scan_op, initial_value);
            exclusive_output = ShuffleUp(inclusive_output, compute::warp_lane_id(), 1u);
            $if(compute::warp_lane_id() == 0)
            {
                exclusive_output = initial_value;
            };
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive