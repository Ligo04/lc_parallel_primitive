/*
 * @Author: Ligo 
 * @Date: 2025-10-22 11:24:49 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 15:23:01
 */
#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/runtime/core.h>


namespace luisa::parallel_primitive
{


using namespace luisa::compute;

// Decoupled look-back(warp)

template <typename T, typename ScanOpT, typename ScanTileStateT>
class TilePrefixCallbackOp : public LuisaModule
{
};

}  // namespace luisa::parallel_primitive