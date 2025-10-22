/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:19:18 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 11:39:13
 */
#pragma once
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{
class WarpExchange : public LuisaModule
{
    // warp level Exchange
  public:
    WarpExchange()  = default;
    ~WarpExchange() = default;

    template <NumericT Type4Byte>
    void Exchange(Var<Type4Byte> d_in)
    {
    }
};
}  // namespace luisa::parallel_primitive