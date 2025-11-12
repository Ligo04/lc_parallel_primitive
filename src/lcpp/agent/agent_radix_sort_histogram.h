/*
 * @Author: Ligo 
 * @Date: 2025-11-12 15:04:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 00:15:05
 */

#pragma once
#include "luisa/dsl/struct.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/stmt.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/common/grid_even_shared.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lcpp/agent/radix_rank_sort_operations.h>
#include <lcpp/block/block_load.h>


namespace luisa::parallel_primitive
{
template <size_t BlockThreads, size_t ItemsPerThread, size_t NOMINAL_4B_NUM_PARTS, typename ComputeT, size_t RadixBits>
struct AgentRadixSortHistogramPolicy
{
    static constexpr size_t BLOCK_THREADS    = BlockThreads;
    static constexpr size_t ITEMS_PER_THREAD = ItemsPerThread;

    static constexpr size_t NUM_PARTS =
        std::max(size_t{1u}, NOMINAL_4B_NUM_PARTS * 4u / std::max(size_t{sizeof(ComputeT)}, size_t{4u}));
    static constexpr size_t RADIX_BITS = RadixBits;
};

namespace details
{
    using namespace luisa::compute;
    template <NumericT KeyType, bool IS_DESCENDING, size_t RADIX_BITS, size_t NUM_PARTS, size_t BLOCK_SIZE, size_t WARP_SIZE, size_t ITEMS_PER_THREAD>
    class AgentRadixSortHistogram : public LuisaModule
    {
      public:
        static constexpr uint TILE_ITEMS      = BLOCK_SIZE * ITEMS_PER_THREAD;
        static constexpr uint RADIX_DIGITS    = 1 << RADIX_BITS;
        static constexpr uint MAX_NUM_PASSES  = (sizeof(KeyType) * 8 + RADIX_BITS - 1) / RADIX_BITS;
        static constexpr uint SHARED_MEM_SIZE = RADIX_DIGITS * NUM_PARTS * MAX_NUM_PASSES;

        using traits                 = radix::traits_t<KeyType>;
        using bit_ordered_type       = typename traits::bit_ordered_type;
        using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

        using Twiddle             = RadixSortTwiddle<IS_DESCENDING, KeyType>;
        using ShmemCounterT       = uint;
        using ShmemAtomicCounterT = ShmemCounterT;

        using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyType>;
        using digit_extractor_t = traits::template digit_extractor_t<fundamental_digit_extractor_t>;


        AgentRadixSortHistogram(SmemTypePtr<uint>   shared_bins,
                                BufferVar<uint>&    bins_out,
                                BufferVar<KeyType>& keys_in,
                                UInt                num_items,
                                UInt                begin_bit,
                                UInt                end_bit)
            : m_shared_bins(shared_bins)
            , d_bins_out(bins_out)
            , d_keys_in(keys_in)
            , num_items(num_items)
            , begin_bit(begin_bit)
            , end_bit(end_bit)
            , num_passes((end_bit - begin_bit + UInt(RADIX_BITS - 1)) / UInt(RADIX_BITS)) {};


        void Init()
        {

            // Initialize shared memory counters to zero
            $for(bin, thread_id().x, UInt(RADIX_DIGITS), UInt(BLOCK_SIZE))
            {
                $for(pass, UInt(0), num_passes)
                {
                    for(auto part = 0u; part < NUM_PARTS; ++part)
                    {
                        UInt index = pass * UInt(RADIX_DIGITS) * UInt(NUM_PARTS)
                                     + bin * UInt(NUM_PARTS) + UInt(part);
                        m_shared_bins->write(index, 0u);
                    }
                };
            };
            sync_block();
        }

        void LoadTileKeys(UInt tile_offset, ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
        {
            Bool full_tile = (num_items - tile_offset >= UInt(TILE_ITEMS));
            $if(full_tile)
            {
                // load direct striped
                LoadDirectStriped<BLOCK_SIZE, bit_ordered_type, ITEMS_PER_THREAD>(
                    thread_id().x, d_keys_in, tile_offset, keys);
            }
            $else
            {
                LoadDirectStriped<BLOCK_SIZE, bit_ordered_type, ITEMS_PER_THREAD>(
                    thread_id().x, d_keys_in, tile_offset, keys, num_items - tile_offset, Twiddle::DefaultKey());
            };

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                keys[i] = Twiddle::In(keys[i]);
            }
        }

        void AccumulateSharedHistograms(UInt tile_offset, const ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
        {
            UInt part = warp_lane_id() % UInt(NUM_PARTS);

            UInt pass = 0;
            $for(current_bit, 0u, end_bit, UInt(RADIX_BITS))
            {
                UInt num_bits = compute::min(+UInt(RADIX_BITS), end_bit - current_bit);

                for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                {
                    UInt bin = digit_extractor_t(current_bit, num_bits).Digit(keys[i]);
                    UInt index = pass * UInt(RADIX_DIGITS) * UInt(NUM_PARTS) + bin * UInt(NUM_PARTS) + part;
                    m_shared_bins->atomic(index).fetch_add(1u);
                }
            };
        }

        void AccumulateGlobalHistograms()
        {
            // Write back shared memory histograms to global memory
            $for(bin, thread_id().x, UInt(RADIX_DIGITS), UInt(BLOCK_SIZE))
            {
                $for(pass, UInt(0), num_passes)
                {
                    ArrayVar<UInt, NUM_PARTS> local_counts;
                    for(auto part = 0u; part < NUM_PARTS; ++part)
                    {
                        UInt index = pass * UInt(RADIX_DIGITS) * UInt(NUM_PARTS)
                                     + bin * UInt(NUM_PARTS) + UInt(part);
                        local_counts[part] = m_shared_bins->read(index);
                    }


                    UInt count = ThreadReduce<uint, NUM_PARTS>().Reduce(
                        local_counts, [](const UInt& a, const UInt& b) { return a + b; });

                    $if(count > 0u)
                    {
                        d_bins_out.atomic(pass * UInt(RADIX_DIGITS) + bin).fetch_add(count);
                    };
                };
            };
        }


        void Process()
        {
            //  avoid overflow uint32 counter
            constexpr uint MAX_PORTION_SIZE = 1 << 30;
            UInt           num_portions     = ceil_div(num_items, UInt(MAX_PORTION_SIZE));

            $for(portion_id, UInt(0), num_portions)
            {
                Init();
                UInt portion_offset = portion_id * MAX_PORTION_SIZE;
                UInt portion_size   = min(num_items - portion_offset, UInt(MAX_PORTION_SIZE));
                $for(offset,
                     block_id().x * UInt(TILE_ITEMS),
                     UInt(portion_size),
                     UInt(TILE_ITEMS) * block_size().x)
                {
                    UInt tile_offset = portion_offset + offset;
                    ArrayVar<bit_ordered_type, ITEMS_PER_THREAD> keys;
                    LoadTileKeys(tile_offset, keys);
                    AccumulateSharedHistograms(tile_offset, keys);
                };
                sync_block();

                // Accumulate to global histogram
                AccumulateGlobalHistograms();
                sync_block();
            };
        }

      private:
        SmemTypePtr<uint> m_shared_bins;

        BufferVar<uint>&    d_bins_out;
        BufferVar<KeyType>& d_keys_in;

        UInt num_items;
        UInt begin_bit, end_bit;
        UInt num_passes;
    };
};  // namespace details
}  // namespace luisa::parallel_primitive


// #define LUISA_RADIX_SORT_HISTOGRAM_TEMSTORAGE_TEMPLATE()                                           \
//     template <NumericT ShmemAtomicCounterT, size_t MAX_NUM_PASSES, size_t RADIX_DIGITS, size_t NUM_PARTS>

// #define LUISA_RADIX_SORT_HISTOGRAM_TEMSTORAGE_TYPE()                                               \
//     luisa::parallel_primitive::details::AgentRadixSortTempStorage<ShmemAtomicCounterT, MAX_NUM_PASSES, RADIX_DIGITS, NUM_PARTS>

// LUISA_TEMPLATE_STRUCT(LUISA_RADIX_SORT_HISTOGRAM_TEMSTORAGE_TEMPLATE,
//                       LUISA_RADIX_SORT_HISTOGRAM_TEMSTORAGE_TYPE,
//                       bin){};
