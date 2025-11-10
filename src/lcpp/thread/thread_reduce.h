/*
 * @Author: Ligo 
 * @Date: 2025-09-22 10:51:36 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 17:06:12
 */

#pragma once
#include <cstddef>
#include <luisa/dsl/var.h>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{

template <typename Type4Byte, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class ThreadReduce : public LuisaModule
{
  public:
    ThreadReduce()  = default;
    ~ThreadReduce() = default;

  public:
    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input, ReduceOp op)
    {
        Var<Type4Byte> result = input[0];
        for(auto i = 1; i < ITEMS_PER_THREAD; ++i)
        {
            result = op(result, input[i]);
        };
        return result;
    };

    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                          ReduceOp                                              op,
                          const Var<Type4Byte>&                                 prefix)
    {
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD + 1> temp;
        temp[0] = prefix;
        for(auto i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            temp[i + 1] = input[i];
        };
        return ThreadReduce<Type4Byte, ITEMS_PER_THREAD + 1>().Reduce(temp, op);
    };
};
}  // namespace luisa::parallel_primitive