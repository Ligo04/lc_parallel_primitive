/*
 * @Author: Ligo 
 * @Date: 2025-10-16 10:59:45 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 18:17:24
 */


#pragma once
#include <luisa/dsl/var.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
template <typename Type4Byte, size_t ITEMS_PER_THREAD = 1>
class ThreadScan : public LuisaModule
{
  public:
    ThreadScan()  = default;
    ~ThreadScan() = default;

    template <typename ScanOp>
    Var<Type4Byte> ThreadScanExclusive(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output,
                                       ScanOp         scan_op,
                                       Var<Type4Byte> prefix,
                                       compute::Bool  apply_prefix = true)
    {
        Var<Type4Byte> inclusive = input[0];
        $if(apply_prefix)
        {
            inclusive = scan_op(prefix, input[0]);
        };

        output[0] = prefix;

        Var<Type4Byte> exclusive = inclusive;

        return ThreadScanExclusive(input, output, inclusive, exclusive, scan_op, 1u);
    };


    template <typename ScanOp>
    Var<Type4Byte> ThreadScanExclusivePartial(
        const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
        ScanOp                                                scan_op,
        compute::UInt                                         valid_items,
        Var<Type4Byte>                                        prefix,
        compute::Bool apply_prefix = true)
    {
        Var<Type4Byte> inclusive = input[0];
        $if(valid_items > 0 & apply_prefix)
        {
            inclusive = scan_op(prefix, input[0]);
        };

        output[0] = prefix;

        Var<Type4Byte> exclusive = inclusive;

        return ThreadScanExclusive(input, output, inclusive, exclusive, scan_op, 1u);
    };

    template <typename ScanOp>
    Var<Type4Byte> ThreadScanInclusive(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output,
                                       ScanOp         scan_op,
                                       Var<Type4Byte> prefix,
                                       compute::Bool  apply_prefix = true)
    {
        Var<Type4Byte> inclusive = input[0];
        $if(apply_prefix)
        {
            inclusive = scan_op(prefix, inclusive);
        };
        output[0] = inclusive;

        return ThreadScanInclusive(input, output, inclusive, scan_op, 1u);
    };

  private:
    template <typename ScanOp>
    Var<Type4Byte> ThreadScanExclusive(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output,
                                       Var<Type4Byte> inclusive,
                                       Var<Type4Byte> exclusive,
                                       ScanOp         scan_op,
                                       compute::UInt  start_index = 1u)
    {
        $for(i, start_index, compute::UInt(ITEMS_PER_THREAD))
        {
            inclusive = scan_op(exclusive, input[i]);

            output[i] = exclusive;

            exclusive = inclusive;
        };
        return inclusive;
    };

    template <typename ScanOp>
    Var<Type4Byte> ThreadScanExclusivePartial(
        const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
        Var<Type4Byte>                                        inclusive,
        Var<Type4Byte>                                        exclusive,
        ScanOp                                                scan_op,
        compute::UInt                                         valid_items,
        compute::UInt                                         start_index = 1u)
    {
        $for(i, start_index, compute::UInt(ITEMS_PER_THREAD))
        {
            $if(i < valid_items)
            {
                inclusive = scan_op(exclusive, input[i]);

                output[i] = exclusive;

                exclusive = inclusive;
            };
        };
        return inclusive;
    };

    template <typename ScanOp>
    Var<Type4Byte> ThreadScanInclusive(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output,
                                       Var<Type4Byte> inclusive,
                                       ScanOp         scan_op,
                                       compute::UInt  start_index = 1u)
    {
        $for(i, start_index, compute::UInt(ITEMS_PER_THREAD))
        {
            inclusive = scan_op(inclusive, input[i]);

            output[i] = inclusive;
        };
        return inclusive;
    };
};

}  // namespace luisa::parallel_primitive