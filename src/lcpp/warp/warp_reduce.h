/*
 * @Author: Ligo 
 * @Date: 2025-09-29 10:43:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-17 17:51:08
 */

#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/warp/details/warp_reduce_shlf.h>

namespace luisa::parallel_primitive
{
enum class WarpReduceAlgorithm
{
    WARP_SHUFFLE       = 0,
    WARP_SHARED_MEMORY = 1
};

template <typename Type4Byte, size_t WARP_SIZE = 32, WarpReduceAlgorithm WarpReduceMethod = WarpReduceAlgorithm::WARP_SHUFFLE>
class WarpReduce : public LuisaModule
{
  public:
    WarpReduce()
    {
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{WARP_SIZE};
        };
    }
    WarpReduce(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }
    ~WarpReduce() = default;

  public:
    // only support power of 2 warp size
    // and only lane_id == 0 will get the correct result
    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& d_in, ReduceOp op, compute::UInt valid_item = WARP_SIZE)
    {
        compute::set_warp_size(WARP_SIZE);
        Var<Type4Byte> result;
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHUFFLE)
        {
            result = details::WarpReduceShfl<Type4Byte, WARP_SIZE>().Reduce(d_in, op, valid_item);
        };
        return result;
    }

    Var<Type4Byte> Sum(const Var<Type4Byte>& lane_value, compute::UInt valid_item = WARP_SIZE)
    {
        return Reduce(
            lane_value,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
            { return a + b; },
            valid_item);
    }

    Var<Type4Byte> Min(const Var<Type4Byte>& lane_value, compute::UInt valid_item = WARP_SIZE)
    {
        return Reduce(
            lane_value,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
            { return compute::min(a, b); },
            valid_item);
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& lane_value, compute::UInt valid_item = WARP_SIZE)
    {
        return Reduce(
            lane_value,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
            { return compute::max(a, b); },
            valid_item);
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive