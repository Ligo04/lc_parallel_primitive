/*
 * @Author: Ligo 
 * @Date: 2025-10-20 16:29:10 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 17:16:51
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/keyvaluepair.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;


struct ArgMaxOp
{
    template <NumericT Type4Byte>
    Var<IndexValuePairT<Type4Byte>> operator()(const Var<IndexValuePairT<Type4Byte>>& a,
                                               const Var<IndexValuePairT<Type4Byte>>& b) const noexcept
    {
        Var<IndexValuePairT<Type4Byte>> result = a;
        $if(b.value > a.value | (b.value == a.value & b.key > a.key))
        {
            result = b;
        };
        return result;
    }
};

struct ArgMinOp
{
    template <NumericT Type4Byte>
    Var<IndexValuePairT<Type4Byte>> operator()(const Var<IndexValuePairT<Type4Byte>>& a,
                                               const Var<IndexValuePairT<Type4Byte>>& b) const noexcept
    {
        Var<IndexValuePairT<Type4Byte>> result = a;
        $if(b.value < a.value | (b.value == a.value & b.key < a.key))
        {
            result = b;
        };
        return result;
    }
};

struct IdentityOp
{
    template <typename TypeData>
    Var<TypeData> operator()(const Var<TypeData>& data) const noexcept
    {
        return data;
    }
};

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

template <typename ScanOp>
struct SwizzleScanOp
{
    ScanOp scan_op;

  public:
    SwizzleScanOp(ScanOp scan_op)
        : scan_op(scan_op)
    {
    }

    template <typename Type>
    Var<Type> operator()(const Var<Type>& a, const Var<Type>& b) const noexcept
    {
        Var<Type> _a = a;
        Var<Type> _b = b;
        return scan_op(_b, _a);
    }
};
};  // namespace luisa::parallel_primitive