/*
 * @Author: Ligo 
 * @Date: 2025-11-10 16:01:44 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 15:38:07
 */


#pragma once

#include <luisa/core/basic_traits.h>
#include <algorithm>
namespace luisa::parallel_primitive
{
// shared memory per block limit 48KB
static constexpr uint max_smem_per_block = 48 * 1024;

template <uint Nominal4ByteBlockThreads, uint Nominal4ByteItemsPerThread, typename T>
struct RegBoundScaling
{
    static constexpr int ITEMS_PER_THREAD =
        std::max(1u, Nominal4ByteItemsPerThread * 4u / std::max(4u, uint{sizeof(T)}));
    static constexpr int BLOCK_THREADS =
        std::min(Nominal4ByteBlockThreads,
                 ceil_div(uint{max_smem_per_block} / (uint{sizeof(T)} * ITEMS_PER_THREAD), 32u) * 32u);
};

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
    static constexpr uint ONESWEEP_RADIX_BITS = 8;
};

/// Epoch-3.5: the tuned default tile (RADIX_ITEMS_PER_THREAD = 16) is sized for
/// <=4-byte keys.  Wider key types scale the per-block shared footprint
/// linearly, so the nominal items-per-thread is scaled down to keep the
/// one-sweep block shared usage under the 32 KiB HLSL/DX codegen floor (and
/// clear of the tighter smem budgets on other backends).  Callers that
/// explicitly instantiate a policy are not affected -- this only adjusts the
/// *effective* tile derived from a nominal template argument.
template <size_t NominalItems, typename KeyType>
static constexpr size_t radix_effective_items_v =
    (sizeof(KeyType) <= 4u)
        ? NominalItems
        : ((NominalItems * 4u / sizeof(KeyType)) < 4u ? 4u : (NominalItems * 4u / sizeof(KeyType)));


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