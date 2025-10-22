/*
 * @Author: Ligo 
 * @Date: 2025-09-22 10:51:36 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 18:17:32
 */

#pragma once


#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{

template <typename Type4Byte>
class ThreadReduce : public LuisaModule
{
    // Implementation details for ThrustReduce
  public:
    ThreadReduce()  = default;
    ~ThreadReduce() = default;

  public:
    template <size_t ITEMS_PER_THREAD = 1, typename ReduceOp>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input, ReduceOp op)
    {
        Var<Type4Byte> result = input[0];
        $for(i, 1u, compute::UInt(ITEMS_PER_THREAD))
        {
            result = op(result, input[i]);
        };
        return result;
    };

    template <size_t ITEMS_PER_THREAD = 1, typename ReduceOp>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                          ReduceOp  op,
                          Type4Byte prefix)
    {
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD + 1> temp;
        temp[0] = prefix;
        $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        {
            temp[i + 1] = input[i];
        };
        return Reduce<ITEMS_PER_THREAD + 1>(temp, op);
    };
};
}  // namespace luisa::parallel_primitive