/*
 * @Author: Ligo 
 * @Date: 2025-09-29 11:30:37 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:36:10
 */
#pragma once

#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>

namespace luisa::parallel_primitive
{
template <NumericT Type4Byte>
class WarpScan : public LuisaModule
{
    using ReduceOpCallable = luisa::compute::Callable<luisa::compute::Var<Type4Byte>()>;

  public:
    WarpScan() {}
    ~WarpScan() = default;

  private:
    template <typename ScanOp>
    ReduceOpCallable Reduce(const Var<Type4Byte>& d_in, ScanOp op)
    {
        return [&]
        {
            // TODO: implement warp scan with op
        };
    }

    ReduceOpCallable Sum(const Var<Type4Byte>& d_in)
    {
        return [&] { warp_active_sum(d_in); };
    }

    ReduceOpCallable Min(const Var<Type4Byte>& d_in)
    {
        return [&] { warp_active_min(d_in); };
    }

    ReduceOpCallable Max(const Var<Type4Byte>& d_in)
    {
        return [&] { warp_active_max(d_in); };
    }
};
}  // namespace luisa::parallel_primitive