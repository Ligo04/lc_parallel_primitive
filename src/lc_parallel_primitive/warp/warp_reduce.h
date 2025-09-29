/*
 * @Author: Ligo 
 * @Date: 2025-09-29 10:43:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:43:14
 */

#pragma once

#include "luisa/dsl/var.h"
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>

namespace luisa::parallel_primitive
{
template <NumericT Type4Byte>
class WarpReduce : public LuisaModule
{
    using ReduceOpCallable = luisa::compute::Callable<Var<Type4Byte>()>;

  public:
    WarpReduce() {}
    ~WarpReduce() = default;

  public:
    template <typename ReduceOp>
    ReduceOpCallable Reduce(const Var<Type4Byte>& d_in, ReduceOp op)
    {
        return [&]
        {
            using namespace luisa::compute;
            // TODO: implement warp reduce with op
            Var<Type4Byte> result = d_in;
            Int            Offset = 1;
            $for(Offset, 1, 32, Offset *= 2)
            {
                Var<Type4Byte> other = warp_active_bit_xor(result);
                result               = op(result, other);
            };
            return result;
        };
    }

    ReduceOpCallable Sum(const Var<Type4Byte>& d_in)
    {
        return [&] { luisa::compute::warp_active_sum(d_in); };
    }

    ReduceOpCallable Min(const Var<Type4Byte>& d_in)
    {
        return [&] { luisa::compute::warp_active_min(d_in); };
    }

    ReduceOpCallable Max(const Var<Type4Byte>& d_in)
    {
        return [&] { luisa::compute::warp_active_max(d_in); };
    }
};
}  // namespace luisa::parallel_primitive