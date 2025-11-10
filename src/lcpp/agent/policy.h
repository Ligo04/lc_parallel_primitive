/*
 * @Author: Ligo 
 * @Date: 2025-11-10 16:01:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 17:53:12
 */


#pragma once

#include <luisa/core/basic_traits.h>
#include <lcpp/common/utils.h>
#include <algorithm>
namespace luisa::parallel_primitive
{

template <uint Nominal4ByteBlockThreads, uint Nominal4ByteItemsPerThread, typename Type>
struct MemBoundScaling
{
    static constexpr uint ITEMS_PER_THREAD =
        std::max(1u, std::min(Nominal4ByteItemsPerThread * 4u / uint{sizeof(Type)}, Nominal4ByteItemsPerThread * 2u));

    static constexpr uint BLOCK_THREADS =
        std::min(Nominal4ByteBlockThreads,
                 ceil_div(uint{256u / uint((sizeof(Type) * ITEMS_PER_THREAD))}, 32u) * 32u);
};

template <uint BlockThreads, uint WarpThreads, uint Nominal4ByteItemsPerThread, typename ComputeT>
struct AgentWarpReducePolicy
{
    static constexpr uint WARP_THREADS  = WarpThreads;
    static constexpr uint BLOCK_THREADS = BlockThreads;

    static constexpr uint ITEMS_PER_THREAD =
        MemBoundScaling<0, Nominal4ByteItemsPerThread, ComputeT>::ITEMS_PER_THREAD;

    static constexpr uint ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;

    static constexpr uint SEGMENTS_PER_BLOCK = BLOCK_THREADS / WARP_THREADS;

    static_assert((BLOCK_THREADS % WARP_THREADS) == 0, "Block should be multiple of warp");
};

template <typename Type>
struct Policy_hub
{
  private:
    static constexpr int items_per_vec_load = 4;

    static constexpr int small_threads_per_warp  = 1;
    static constexpr int medium_threads_per_warp = 32;

    static constexpr int nominal_4b_large_threads_per_block = 256;

    static constexpr int nominal_4b_small_items_per_thread  = 16;
    static constexpr int nominal_4b_medium_items_per_thread = 16;
    static constexpr int nominal_4b_large_items_per_thread  = 16;

  public:
    using SmallReducePolicy =
        AgentWarpReducePolicy<256, small_threads_per_warp, nominal_4b_small_items_per_thread, Type>;
    using MediumReducePolicy =
        AgentWarpReducePolicy<256, medium_threads_per_warp, nominal_4b_medium_items_per_thread, Type>;
};
}  // namespace luisa::parallel_primitive