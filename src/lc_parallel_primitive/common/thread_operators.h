/*
 * @Author: Ligo 
 * @Date: 2025-10-20 16:29:10 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 22:35:38
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>


namespace luisa::parallel_primitive
{
using namespace luisa::compute;
template <typename ReduceOpT>
struct ReduceBySegmentOp
{
    ReduceOpT reduce_op;

    template <KeyValuePairType KeyValuePairT>
    Var<KeyValuePairT> operator()(const Var<KeyValuePairT>& a,
                                  const Var<KeyValuePairT>& b) const noexcept
    {
        Var<KeyValuePairT> result;
        result.key = a.key + b.key;
        $if(b.key > 0)
        {
            result.value = b.value;
        }
        $else
        {
            result.value = reduce_op(a.value, b.value);
        };
        return result;
    }
};


template <typename ReduceOpT>
struct ReduceByKeyOp
{
    ReduceOpT reduce_op;

    template <KeyValuePairType KeyValuePairT>
    Var<KeyValuePairT> operator()(const Var<KeyValuePairT>& a,
                                  const Var<KeyValuePairT>& b) const noexcept
    {
        Var<KeyValuePairT> result = b;
        $if(a.key == b.key)
        {
            result.value = reduce_op(a.value, b.value);
        };
        return result;
    }
};
};  // namespace luisa::parallel_primitive