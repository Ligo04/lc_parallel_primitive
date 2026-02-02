/*
 * @Author: Ligo
 * @Date: 2025-11-13 16:31:28
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-14 11:23:14
 */
#pragma once
#include <array>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/core/basic_traits.h>
#include <lcpp/block/detail/block_scan_warp.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/thread/thread_scan.h>
#include <lcpp/block/block_scan.h>

namespace luisa::parallel_primitive
{
enum class RadixRankAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};
using namespace luisa::compute;
// template <uint               RadixBits,
//           bool               IsDescending,
//           bool               MemoizeOuterScan  = true,
//           size_t             BLOCK_SIZE        = details::BLOCK_SIZE,
//           size_t             ITEMS_PER_THREAD  = details::ITEMS_PER_THREAD,
//           size_t             WARP_SIZE         = details::WARP_SIZE,
//           RadixRankAlgorithm DEFAULT_ALGORITHM = RadixRankAlgorithm::WARP_SHUFFLE>
// class BlockRadixRank : public LuisaModule
// {
//   private:
//     using DigitCounter = ushort;
//     // Integer type for packing DigitCounters into columns of shared memory banks
//     using PackedCounter = uint;

//     static constexpr DigitCounter max_tile_size = std::numeric_limits<DigitCounter>::max();

//     static constexpr uint RADIX_DIGITS = 1 << RadixBits;

//     static constexpr uint WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

//     static constexpr uint BYTES_PER_COUNTER = uint(sizeof(DigitCounter));
//     // static constexpr uint LOG_BYTES_PER_COUNTER =
//     // Log2<BYTES_PER_COUNTER>::VALUE;

//     static constexpr uint PACKING_RATIO = static_cast<uint>(sizeof(PackedCounter) / sizeof(DigitCounter));
//     static constexpr uint LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE;

//     // Always at least one lane
//     static constexpr uint LOG_COUNTER_LANES = std::max(RadixBits - LOG_PACKING_RATIO, 0);
//     static constexpr uint COUNTER_LANES     = 1 << LOG_COUNTER_LANES;

//     // The number of packed counters per thread (plus one for padding)
//     static constexpr uint PADDED_COUNTER_LANES = COUNTER_LANES + 1;
//     static constexpr uint RAKING_SEGMENT       = PADDED_COUNTER_LANES;

//     static_assert(PADDED_COUNTER_LANES * PACKING_RATIO == RAKING_SEGMENT,
//                   "PADDL_COUNTER_LANES * PACKING_RATIO = RAKING_SEGMENT");

//     struct PrefixCallBack
//     {
//         Var<PackedCounter> operator()(const Var<PackedCounter>& block_aggregate)
//         {
//             Var<PackedCounter> block_prefix = 0;

//             // Propagate totals in packed fields
//             for(auto PACKED = 1u; PACKED < PACKING_RATIO; PACKED++)
//             {
//                 block_prefix += block_aggregate << Var<PackedCounter>((sizeof(DigitCounter) * 8 * PACKED));
//             }

//             return block_prefix;
//         }
//     };

//     static Callable pack_digit_counters = [](const ArrayVar<DigitCounter, PACKING_RATIO>& dc)
//     {
//         static_assert(sizeof(PackedCounter) == sizeof(DigitCounter) * PACKING_RATIO, "PACKING_RATIO mismatch.");

//         Var<PackedCounter> p = 0u;
//         for(uint i = 0; i < PACKING_RATIO; i++)
//         {
//             p |= (Var<PackedCounter>(dc[i]) << Var<PackedCounter>(8u * sizeof(DigitCounter) * i));
//         }
//         return p;
//     };


//     static Callable unpack_digit_counters = [](Var<PackedCounter> p, ArrayVar<DigitCounter, PACKING_RATIO>& dc)
//     {
//         for(uint i = 0; i < PACKING_RATIO; i++)
//         {
//             dc[i] = Var<DigitCounter>((p >> (8u * sizeof(DigitCounter) * i))
//                                       & ((Var<DigitCounter>(1) << (8u * sizeof(DigitCounter))) - 1));
//         }
//     };

//     static Callable digit_index = [](UInt i, UInt j, UInt k)
//     { return i * UInt(BLOCK_SIZE * PACKING_RATIO) + j * UInt(PACKING_RATIO) + k; };

//     static Callable packed_index = [](UInt i, UInt j) { return i * UInt(BLOCK_SIZE) + j; };

//   public:
//     static constexpr uint BINS_TRACKED_PER_THREAD = std::max(1, (RADIX_DIGITS + BLOCK_SIZE - 1) / BLOCK_SIZE);

//   public:
//     BlockRadixRank()
//     {
//         // Allocate shared memory for digit counters
//         m_shared_digit_counters = new SmemType<DigitCounter>{PADDED_COUNTER_LANES * BLOCK_SIZE * PACKING_RATIO};
//         // row major layout
//         m_linear_tid = thread_id().z * (block_size_x() * block_size_y())
//                        + thread_id().y * block_size_x() + thread_id().x;
//     }
//     ~BlockRadixRank() = default;

//   public:
//     template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
//     void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
//                   compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
//                   DigitExtractorT                                        digit_extractor)
//     {
//         static_assert(BLOCK_SIZE * KEY_PER_THREAD <= max_tile_size,
//                       "DigitCounter type is too small to hold this number of keys.");

//         ArrayVar<DigitCounter, KEY_PER_THREAD> thread_prefixes;
//         ArrayVar<DigitCounter, KEY_PER_THREAD> digit_counters;

//         ResetCounters();

//         for(auto item = 0u; item < KEY_PER_THREAD; ++item)
//         {
//             UInt digit = digit_extractor.Digit(keys[item]);

//             UInt sub_counter = digit >> LOG_COUNTER_LANES;

//             UInt counter_lane = digit & (COUNTER_LANES - 1);

//             if constexpr(IsDescending)
//             {
//                 sub_counter  = (RADIX_DIGITS - 1) - sub_counter;
//                 counter_lane = (COUNTER_LANES - 1) - counter_lane;
//             }

//             // Pointer to smem digit counter
//             digit_counters =
//                 m_shared_digit_counters->read(digit_index(counter_lane, m_linear_tid, sub_counter));

//             thread_prefixes[item] = digit_counters[item];
//             digit_counters[item]  = thread_prefixes[item] + 1;
//         }

//         sync_block();
//         ScanCounters();
//         sync_block();

//         // Extract the local ranks of each key
//         for(auto item = 0u; item < KEY_PER_THREAD; ++item)
//         {
//             // Add in thread block exclusive prefix
//             ranks[item] = thread_prefixes[item] + digit_counters[item];
//         }
//     }

//     template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
//     void RankKeys(const compute::ArrayVar<UnsignedBits, ITEMS_PER_THREAD>& keys,
//                   compute::ArrayVar<uint, ITEMS_PER_THREAD>&               ranks,
//                   DigitExtractorT                                          digit_extractor,
//                   compute::ArrayVar<uint, ITEMS_PER_THREAD>&               exclusive_digit_prefix)
//     {
//     }

//   private:
//     void ResetCounters()
//     {
//         // Reset shared memory digit counters
//         for(auto LANE = 0u; LANE < PADDED_COUNTER_LANES; ++LANE)
//         {
//             for(auto PACKED = 0u; PACKED < PACKING_RATIO; ++PACKED)
//             {
//                 UInt offset = digit_index(LANE, m_linear_tid, PACKED);
//                 m_shared_digit_counters->write(offset, 0u);
//             }
//         }
//     }

//     void UpSweep()
//     {
//         ArrayVar<PackedCounter, RAKING_SEGMENT> thread_counters;

//         for(auto i = 0u; i < RAKING_SEGMENT; ++i)
//         {
//             ArrayVar<DigitCounter, PACKING_RATIO> dc;
//             for(auto j = 0u; j < PACKING_RATIO; ++j)
//             {
//                 dc[j] = m_shared_digit_counters->read(digit_index(i, m_linear_tid, j));
//             }

//             thread_counters[i] = pack_digit_counters(dc);
//         }

//         return ThreadReduce<PackedCounter, RAKING_SEGMENT>().Reduce(
//             thread_counters,
//             [](const Var<PackedCounter>& a, const Var<PackedCounter>& b) { return a + b; });
//     }

//     void ExclusiveDownsweep(Var<PackedCounter>& exclusive_partial)
//     {

//         ArrayVar<PackedCounter, RAKING_SEGMENT> smem_ranking;
//         // details::Thread
//     }

//     void ScanCounters()
//     {
//         Var<PackedCounter> raking_partial = UpSweep();
//         Var<PackedCounter> exclusive_partial;
//         PrefixCallBack     prefix_call_back;
//         BlockScan<PackedCounter, BLOCK_SIZE, 1, WARP_SIZE>().ExclusiveSum(raking_partial, exclusive_partial, prefix_call_back);

//         // Downsweep scan with exclusive partial
//         ExclusiveDownsweep(exclusive_partial);
//     }


//   private:
//     // DigitCounter        digit_counters[PADDED_COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
//     // PackedCounter       raking_grid[BLOCK_THREADS][RAKING_SEGMENT];
//     // PADDEL_COUNTER_LANES * PACKING_RATIO = RAKING_SEGMENT
//     SmemTypePtr<DigitCounter> m_shared_digit_counters;
//     /// Linear thread-id
//     UInt m_linear_tid;
// };


template <uint BLOCK_THREADS, uint RadixBits, bool IsDescending, WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, int NUM_PARTS = 1, uint WARP_SIZE = details::WARP_SIZE>
class BlockRadixRankMatchEarlyCounts : public LuisaModule
{
    static constexpr int RADIX_DIGITS    = 1 << RadixBits;
    static constexpr int BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
    static constexpr int BINS_TRACKED_PER_THREAD = BINS_PER_THREAD;
    static constexpr int FULL_BINS               = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS;
    static constexpr int PARTIAL_WARP_THREADS    = BLOCK_THREADS % WARP_SIZE;
    static constexpr int BLOCK_WARPS             = BLOCK_THREADS / WARP_SIZE;
    static constexpr int PARTIAL_WARP_ID         = BLOCK_WARPS - 1;
    static constexpr int WARP_MASK               = ~0;
    static constexpr int NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0;
    static constexpr int MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS;

    // types
    // using BlockScan = cub::BlockScan<int, BLOCK_THREADS, InnerScanAlgorithm>;

  public:
    BlockRadixRankMatchEarlyCounts() {}
    ~BlockRadixRankMatchEarlyCounts() = default;

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        // union
        // {
        //     int warp_offsets[BLOCK_WARPS][RADIX_DIGITS];
        //     int warp_histograms[BLOCK_WARPS][RADIX_DIGITS][NUM_PARTS];
        // };
        SmemTypePtr<uint> warp_histograms;
        DigitExtractorT   digit_extractor;
        CountsCallback    counts_callback;
        UInt              warp;
        UInt              lane;

        // Callable<uint> Digit = [](Var<UnsignedBits> key)
        // {
        //     UInt digit = digit_extractor.Digit(key);
        //     return IsDescending ? RADIX_DIGITS - 1 - digit : digit;
        // };

        inline static Callable ThreadBin = [](UInt u)
        {
            UInt bin = thread_id().x * BINS_PER_THREAD + u;
            return IsDescending ? RADIX_DIGITS - 1 - bin : bin;
        };

        BlockRadixRankMatchInternal(DigitExtractorT digit_extractor, CountsCallback callback)
            : digit_extractor(digit_extractor)
            , counts_callback(callback)
            , warp(thread_id().x / UInt(WARP_SIZE))
            , lane(warp_lane_id())
        {
        }

        void ComputeHistogramWarp(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys)
        {
            // ArrayVar<uint, RADIX_DIGITS * NUM_PARTS> warp_histograms;
            $for(bin, lane, UInt(RADIX_DIGITS), UInt(WARP_SIZE))
            {
                for(auto part = 0u; part < NUM_PARTS; ++part)
                {
                    warp_histograms->write(warp * UInt(RADIX_DIGITS) * UInt(NUM_PARTS)
                                               + bin * UInt(NUM_PARTS) + UInt(part),
                                           0u);
                }
            };

            // TODO: sync_warp();
            sync_block();

            for(auto i = 0u; i < KEY_PER_THREAD; ++i)
            {
                UInt bin   = Digit(keys[i]);
                UInt index = warp * UInt(RADIX_DIGITS) * UInt(NUM_PARTS) + bin * UInt(NUM_PARTS)
                             + (lane % UInt(NUM_PARTS));
                warp_histograms->atomic(index).fetch_add(1u);
            }
        }

        void RankKeys(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                      ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                      ArrayVar<uint, BINS_TRACKED_PER_THREAD>&      exclusive_digit_prefix)
        {
            // TODO implement radix rank with early counts
        }
    };


  public:
    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_TRACKED_PER_THREAD>&      exclusive_digit_prefix,
                  CountsCallback                                         counts_callback)
    {
        // TODO implement radix rank
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_TRACKED_PER_THREAD>&      exclusive_digit_prefix)
    {
        // TODO implement radix rank with exclusive prefix
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor)
    {
        compute::ArrayVar<uint, BINS_TRACKED_PER_THREAD> exclusive_digit_prefix;
        RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};
}  // namespace luisa::parallel_primitive