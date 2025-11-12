/*
 * @Author: Ligo 
 * @Date: 2025-11-12 14:42:25 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 16:43:49
 */
#pragma once

#include <algorithm>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <limits>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/utils.h>
#include <lcpp/agent/policy.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceDeviceHistogram : public LuisaModule
{
};
}  // namespace luisa::parallel_primitive