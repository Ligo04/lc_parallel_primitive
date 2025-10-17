/*
 * @Author: Ligo 
 * @Date: 2025-09-29 10:43:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-17 10:42:09
 */

#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>

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
    // only land_id == 0 will get the correct result
    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& d_in, ReduceOp op)
    {
        compute::set_warp_size(WARP_SIZE);
        Var<Type4Byte> result = d_in;
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHARED_MEMORY)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            (*m_shared_mem)[land_id] = result;

            // TODO: sync_warp()
        }
        $elif(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHUFFLE)
        {
            compute::UInt land_id   = compute::warp_lane_id();
            compute::UInt wave_size = compute::warp_lane_count();

            Var<Type4Byte> output;
            compute::UInt  offset = 1u;
            $while(offset < wave_size)
            {
                Var<Type4Byte> temp = compute::warp_read_lane(result, land_id + offset);
                $if(land_id + offset < wave_size)
                {
                    result = op(result, temp);
                };
                offset <<= 1;
            };
        };
        return result;
    }

    Var<Type4Byte> Sum(const Var<Type4Byte>& lane_value)
    {
        Var<Type4Byte> result;
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHARED_MEMORY) {}
        $else
        {
            result = Reduce(lane_value,
                            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
                            { return a + b; });
        };
        return result;
    }

    Var<Type4Byte> Min(const Var<Type4Byte>& lane_value)
    {
        Var<Type4Byte> result;
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHARED_MEMORY) {}
        $else
        {
            result = Reduce(lane_value,
                            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
                            { return compute::min(a, b); });
        };
        return result;
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& lane_value)
    {
        Var<Type4Byte> result;
        $if(WarpReduceMethod == WarpReduceAlgorithm::WARP_SHARED_MEMORY) {}
        $else
        {
            result = Reduce(lane_value,
                            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) noexcept
                            { return compute::max(a, b); });
        };
        return result;
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive