/*
 * @Author: Ligo 
 * @Date: 2025-10-14 16:26:29 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-14 16:34:40
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>


namespace luisa::parallel_primitive
{
class BlockExchange : public LuisaModule
{
  public:
    BlockExchange()  = default;
    ~BlockExchange() = default;

  public:
};
}  // namespace luisa::parallel_primitive