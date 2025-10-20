/*
 * @Author: Ligo 
 * @Date: 2025-09-29 11:30:37 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 14:55:53
 */
#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/warp/details/warp_scan_shlf.h>

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
                       const Var<Type4Byte>& initial_value)
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>().ExclusiveScan(
                thread_data, exclusive_output, scan_op, initial_value);
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY){};
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);

        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>().ExclusiveScan(
                thread_data, exclusive_output, warp_aggregate, scan_op, initial_value);
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY){};
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_in, Var<Type4Byte>& inclusive_output, ScanOp op)
    {
        InclusiveScan(thread_in, inclusive_output, op, Type4Byte(0));
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>().InclusiveScan(
                thread_data, inclusive_output, scan_op, initial_value);
        }
        $elif(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE){};
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value = Type4Byte(0))
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>().InclusiveScan(
                thread_data, inclusive_output, warp_aggregate, scan_op, initial_value);
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

    template <typename ScanOp>
    void Scan(const Var<Type4Byte>& thread_data,
              Var<Type4Byte>&       inclusive_output,
              Var<Type4Byte>&       exclusive_output,
              ScanOp                scan_op)
    {
        compute::set_warp_size(WARP_SIZE);
        $if(WarpScanMethod == WarpScanAlgorithm::WARP_SHUFFLE)
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>().Scan(
                thread_data, inclusive_output, exclusive_output, scan_op);
        };
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem = nullptr;
};
}  // namespace luisa::parallel_primitive