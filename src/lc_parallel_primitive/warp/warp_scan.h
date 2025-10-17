/*
 * @Author: Ligo 
 * @Date: 2025-09-29 11:30:37 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-17 11:16:44
 */
#pragma once

#include "luisa/dsl/var.h"
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>

namespace luisa::parallel_primitive
{
enum class WarpScanAlgorithm
{
    WARP_SHUFFLE       = 0,
    WARP_SHARED_MEMORY = 1
};
template <NumericT Type4Byte, size_t WARP_SIZE = 32, WarpScanAlgorithm WarpScanMethod = WarpScanAlgorithm::WARP_SHUFFLE>
class WarpScan : public LuisaModule
{
  public:
    WarpScan()
    {
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{WARP_SIZE};
        };
    }
    WarpScan(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }
    ~WarpScan() = default;

  public:
    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output, ScanOp op)
    {
        ExclusiveScan(thread_data, exclusive_output, op, Type4Byte(0));
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);
        Var<Type4Byte> result = thread_data;
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            // TODO: sync_warp()
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            InclusiveScan(thread_data, result, scan_op, initial_value);
            result = compute::warp_read_lane(result, compute::warp_lane_id() - 1u);
            $if(compute::warp_lane_id() == 0u)
            {
                result = initial_value;
            };
        };
        exclusive_output = result;
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);
        Var<Type4Byte> result;
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            // TODO: sync_warp()
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            InclusiveScan(thread_data, result, warp_aggregate, scan_op, initial_value);
            result = compute::warp_read_lane(result, compute::warp_lane_id() - 1u);
            $if(compute::warp_lane_id() == 0u)
            {
                result = initial_value;
            };
        };
        exclusive_output = result;
    }


    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& d_in, Var<Type4Byte>& out, ScanOp op)
    {
        InclusiveScan(d_in, out, op, Type4Byte(0));
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_input,
                       Var<Type4Byte>&       inclusive_out,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            // TODO: sync_warp()
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            Var<Type4Byte> output = thread_input;
            compute::UInt  offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = compute::warp_read_lane(output, land_id - offset);
                $if(land_id >= offset)
                {
                    output = scan_op(temp, output);
                };
                offset <<= 1;
            };
            inclusive_out = output;
        };
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            // TODO: sync_warp()
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            InclusiveScan(thread_data, inclusive_output, scan_op, initial_value);
            warp_aggregate = compute::warp_read_lane(inclusive_output,
                                                     compute::warp_lane_count() - 1u);
        };
    }

    // sum
    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output)
    {
        ExclusiveScan(
            thread_data,
            exclusive_output,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            Type4Byte(0));
    }

    void ExclusiveSum(const Var<Type4Byte>& thread_data,
                      Var<Type4Byte>&       exclusive_output,
                      Var<Type4Byte>&       warp_aggregate)
    {
        ExclusiveScan(
            thread_data,
            exclusive_output,
            warp_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            Type4Byte(0));
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_output)
    {
        InclusiveScan(
            thread_data,
            inclusive_output,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            Type4Byte(0));
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data,
                      Var<Type4Byte>&       inclusive_output,
                      Var<Type4Byte>&       warp_aggregate)
    {
        InclusiveScan(
            thread_data,
            inclusive_output,
            warp_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            Type4Byte(0));
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem = nullptr;
};
}  // namespace luisa::parallel_primitive