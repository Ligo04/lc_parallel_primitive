/*
 * @Author: Ligo 
 * @Date: 2025-11-10 16:01:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 22:23:19
 */


#pragma once

#include <luisa/core/basic_traits.h>
#include <lcpp/common/utils.h>
#include <algorithm>
namespace luisa::parallel_primitive
{
// shared memory per block limit 48KB
static constexpr uint max_smem_per_block = 48 * 1024;

template <uint Nominal4ByteBlockThreads, uint Nominal4ByteItemsPerThread, typename Type>
struct MemBoundScaling
{
    static constexpr uint ITEMS_PER_THREAD =
        std::max(1u, std::min(uint{Nominal4ByteItemsPerThread * 4u / sizeof(Type)}, Nominal4ByteItemsPerThread * 2u));

    static constexpr uint BLOCK_THREADS =
        std::min(Nominal4ByteBlockThreads,
                 ceil_div(uint{max_smem_per_block / (sizeof(Type) * ITEMS_PER_THREAD)}, 32u) * 32u);
};

template <uint BlockThreads, uint WarpThreads, uint Nominal4ByteItemsPerThread, typename ComputeT>
struct AgentWarpReducePolicy
{
    static constexpr uint WARP_THREADS  = WarpThreads;
    static constexpr uint BLOCK_THREADS = BlockThreads;

    static constexpr uint ITEMS_PER_THREAD =
        MemBoundScaling<0, Nominal4ByteItemsPerThread, ComputeT>::ITEMS_PER_THREAD;

    static constexpr uint ITEMS_PER_TILE = ITEMS_PER_THREAD * WARP_THREADS;

    static constexpr uint SEGMENTS_PER_BLOCK = BLOCK_THREADS / WARP_THREADS;

    static_assert((BLOCK_THREADS % WARP_THREADS) == 0, "Block should be multiple of warp");
};

template <typename Type>
struct Policy_hub
{
  private:
    static constexpr int small_threads_per_warp             = 32;
    static constexpr int nominal_4b_large_threads_per_block = 256;

    static constexpr int nominal_4b_small_items_per_thread = 2;
    static constexpr int nominal_4b_large_items_per_thread = 2;

  public:
    using SmallReducePolicy =
        AgentWarpReducePolicy<nominal_4b_large_threads_per_block, small_threads_per_warp, nominal_4b_small_items_per_thread, Type>;
};


template <typename KeyType>
struct OneSweepSmallKeyTunedPolicy
{
    static constexpr bool ONESWEEP            = true;
    static constexpr uint ONESWEEP_RAIDX_BITS = 8;
};


template <int BlockThreads, int PixelsPerThread, bool RleCompress, bool WorkStealing, int VecSize = 4>
struct AgentHistogramPolicy
{
    /// Threads per thread block
    static constexpr int BLOCK_THREADS = BlockThreads;
    /// Pixels per thread (per tile of input)
    static constexpr int PIXELS_PER_THREAD = PixelsPerThread;

    /// Whether to perform localized RLE to compress samples before histogramming
    static constexpr bool IS_RLE_COMPRESS = RleCompress;

    /// Whether to dequeue tiles from a global work queue
    static constexpr bool IS_WORK_STEALING = WorkStealing;

    /// Vector size for samples loading (1, 2, 4)
    static constexpr int VEC_SIZE = VecSize;
};
}  // namespace luisa::parallel_primitive