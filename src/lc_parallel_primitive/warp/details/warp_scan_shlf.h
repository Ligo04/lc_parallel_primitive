/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:00:46 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-21 22:57:00
 */


#pragma once


#include "luisa/dsl/stmt.h"
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>
#include <lc_parallel_primitive/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT Type4Byte>
    Var<Type4Byte> ShuffleUp(Var<Type4Byte>& input, UInt curr_lane_id, UInt offset, UInt first_lane = 0u)
    {
        Var<Type4Byte> result   = input;
        UInt           src_lane = UInt(curr_lane_id - offset);
        $if(src_lane >= first_lane)
        {
            result = compute::warp_read_lane(input, src_lane);
        };
        return result;
    };

    template <NumericT KeyType, NumericT ValueType>
    Var<KeyValuePair<KeyType, ValueType>> ShuffleUp(Var<KeyValuePair<KeyType, ValueType>>& input,
                                                    UInt curr_lane_id,
                                                    UInt offset,
                                                    UInt first_lane = 0u)
    {
        Var<KeyValuePair<KeyType, ValueType>> result = input;
        UInt src_lane = UInt(curr_lane_id - offset);
        $if(src_lane >= first_lane)
        {
            result.key   = compute::warp_read_lane(input.key, src_lane);
            result.value = compute::warp_read_lane(input.value, src_lane);
        };
        return result;
    };


    template <typename Type4Byte, size_t WARP_SIZE = 32>
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
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           ScanOp                scan_op)
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
                // device_log("thid:{} - lane_id {}, offset {}, temp {}, output {}",
                //            compute::dispatch_id().x,
                //            lane_id,
                //            offset,
                //            temp,
                //            output);
                offset <<= 1;
            };
            inclusive_output = output;
            // device_log("thid:{} - lane_id {}, inclusive_output {},input {}",
            //            compute::dispatch_id().x,
            //            lane_id,
            //            inclusive_output,
            //            thread_input);
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