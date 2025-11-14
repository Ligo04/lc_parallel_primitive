/*
 * @Author: Ligo
 * @Date: 2025-11-12 15:04:20
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-14 13:43:38
 */

#pragma once
#include <cstddef>
#include <lcpp/agent/policy.h>
#include <lcpp/agent/radix_rank_sort_operations.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_radix_rank.h>
#include <lcpp/common/grid_even_shared.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/warp/warp_reduce.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/stmt.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

namespace luisa::parallel_primitive
{

template <uint NominalBlockThreads4B, uint NominalItemsPerThread4B, typename ComputeT, uint RankNumParts, uint RadixBits>
struct AgentRadixSortOneSweepPolicy : RegBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>
{
    static constexpr uint RANK_NUM_PARTS = RankNumParts;
    static constexpr uint RADIX_BITS     = RadixBits;
};

namespace details
{
    using namespace luisa::compute;
    template <NumericT KeyType, NumericT ValueType, size_t RADIX_BITS, size_t RANK_NUM_PARTS, bool KEYS_ONLY, bool IS_DESCENDING, size_t BLOCK_SIZE, size_t WARP_SIZE, size_t ITEMS_PER_THREAD>
    class AgentRadixSortOneSweep : public LuisaModule
    {
      public:
        // constants
        static constexpr uint TILE_ITEMS            = BLOCK_SIZE * ITEMS_PER_THREAD;
        static constexpr uint RADIX_DIGITS          = 1 << RADIX_BITS;
        static constexpr uint BINS_PER_THREAD       = (RADIX_DIGITS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        static constexpr bool FULL_BINS             = BINS_PER_THREAD * BLOCK_SIZE == RADIX_DIGITS;
        static constexpr uint WARP_THREADS          = WARP_SIZE;
        static constexpr uint BLOCK_WARPS           = BLOCK_SIZE / WARP_THREADS;
        static constexpr uint WARP_MASK             = ~0;
        static constexpr uint LOOKBACK_PARTIAL_MASK = 1 << (uint(sizeof(uint)) * 8 - 2);
        static constexpr uint LOOKBACK_GLOBAL_MASK  = 1 << (uint(sizeof(uint)) * 8 - 1);
        static constexpr uint LOOKBACK_KIND_MASK    = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK;
        static constexpr uint LOOKBACK_VALUE_MASK   = ~LOOKBACK_KIND_MASK;

        using traits                 = radix::traits_t<KeyType>;
        using bit_ordered_type       = typename traits::bit_ordered_type;
        using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

        using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyType>;
        using digit_extractor_t = typename traits::template digit_extractor_t<fundamental_digit_extractor_t>;

        using Twiddle = RadixSortTwiddle<IS_DESCENDING, KeyType>;

        using BlockRadixRankT = BlockRadixRankMatchEarlyCounts<BLOCK_SIZE, RADIX_BITS, IS_DESCENDING>;

        static Callable ThreadBin = [](UInt u) { return thread_id().x * UInt(BINS_PER_THREAD) + u; };

        struct CountsCallback
        {
            using RadixSortOneSweepPolicy = AgentRadixSortOneSweepPolicy<1u, 8u, KeyType, 1u, RADIX_BITS>;
            using AgentT =
                AgentRadixSortOneSweep<KeyType, ValueType, RADIX_BITS, RadixSortOneSweepPolicy::RANK_NUM_PARTS, KEYS_ONLY, IS_DESCENDING, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;
            AgentT&                                       agent;
            ArrayVar<uint, BINS_PER_THREAD>&              bins;
            ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys;
            static constexpr bool                         EMPTY = false;
            CountsCallback(AgentT&                                       agent,
                           ArrayVar<uint, BINS_PER_THREAD>&              bins,
                           ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
                : agent(agent)
                , bins(bins)
                , keys(keys)
            {
            }
            void operator()(const ArrayVar<uint, BINS_PER_THREAD>& other_bins)
            {
                for(auto u = 0; u < BINS_PER_THREAD; ++u)
                {
                    bins[u] = other_bins[u];
                }
                agent.LookbackPartial(bins);

                agent.TryShortCircuit(keys, bins);
            }
        };

        AgentRadixSortOneSweep(SmemTypePtr<bit_ordered_type> smem_keys,
                               BufferVar<uint>&              d_bins_out,
                               BufferVar<uint>&              d_ctrs,
                               BufferVar<KeyType>&           keys_in,
                               BufferVar<KeyType>&           keys_out,
                               BufferVar<ValueType>&         values_in,
                               BufferVar<ValueType>&         values_out,
                               UInt                          num_items,
                               UInt                          begin_bit,
                               UInt                          end_bit)
            : m_shared_keys(smem_keys)
            , d_bins_out(d_bins_out)
            , d_ctrs(d_ctrs)
            , d_keys_in(keys_in)
            , d_keys_out(keys_out)
            , d_values_in(values_in)
            , d_values_out(values_out)
            , num_items(num_items)
            , begin_bit(begin_bit)
            , end_bit(end_bit)
        {
            $if(thread_id().x == 0)
            {
                m_block_idx->write(0, d_ctrs.atomic(0u).fetch_add(1u));
            };

            sync_block();
            block_idx  = m_block_idx->read(0);
            full_block = (block_idx + 1) * UInt(TILE_ITEMS) <= num_items;
        }

        void LoadKeys(UInt tile_offset, ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
        {
            ArrayVar<KeyType, ITEMS_PER_THREAD> temp_keys;
            $if(full_block)
            {
                LoadDirectWarpStriped(thread_id().x, d_keys_in, tile_offset, temp_keys);
            }
            $else
            {
                LoadDirectWarpStriped(
                    thread_id().x, d_keys_in, tile_offset, temp_keys, num_items - tile_offset, Twiddle::DefaultKey());
            };

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                keys[i] = Twiddle::In(Var<bit_ordered_type>(temp_keys[i]));
            }
        }

        void LoadValues(UInt tile_offset, ArrayVar<ValueType, ITEMS_PER_THREAD>& values)
        {
            $if(full_block)
            {
                LoadDirectWarpStriped(thread_id().x, d_values_in, tile_offset, values);
            }
            $else
            {
                LoadDirectWarpStriped(thread_id().x, d_values_in, tile_offset, values, num_items - tile_offset);
            };
        }

        void ComputeKeyDigits(ArrayVar<uint, ITEMS_PER_THREAD>& digits)
        {
            for(auto u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                UInt idx  = thread_id().x + u * UInt(BLOCK_SIZE);
                digits[u] = digit_extractor_t().Digit(m_shared_keys->read(idx));
            }
        }

        void GatherScatterValues(const ArrayVar<uint, ITEMS_PER_THREAD>& ranks)
        {
            ArrayVar<uint, ITEMS_PER_THREAD> digits;
            ComputeKeyDigits(digits);

            ArrayVar<ValueType, ITEMS_PER_THREAD> values;
            LoadValues(block_idx * TILE_ITEMS, values);

            sync_block();
            ScatterValuesShared(values, ranks);
            sync_block();
            ScatterValuesGlobal(digits);
        }

        void Process()
        {
            ArrayVar<bit_ordered_type, ITEMS_PER_THREAD> keys;
            LoadKeys(block_idx * TILE_ITEMS, keys);

            ArrayVar<uint, ITEMS_PER_THREAD> ranks;
            ArrayVar<uint, ITEMS_PER_THREAD> exclusive_digit_prefix;
            ArrayVar<uint, ITEMS_PER_THREAD> bins;
            BlockRadixRankT().template RankKeys<KeyType>(
                keys, ranks, digit_extractor_t(), exclusive_digit_prefix, CountsCallback(*this, bins, keys));

            sync_block();
            ScatterKeysShared(keys, ranks);

            LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
            LoadbackGlobal(bins);
            UpdateBinsGlobal(bins, exclusive_digit_prefix);

            sync_block();
            ScatterKeysGlobal();
        }


        void LookbackPartial(const ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                UInt bin = ThreadBin(i);
                $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                {
                    UInt value = bins[i] | UInt(LOOKBACK_PARTIAL_MASK);
                    d_lookback.volatile_write(block_idx * UInt(RADIX_DIGITS) + bin, value);
                };
            }
        }

        void TryShortCircuit(const ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys,
                             const ArrayVar<uint, BINS_PER_THREAD>&              bins)
        {
            // check if any bins can be short-circuited
            Bool short_circuit = true;

            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                $if(FULL_BINS | ThreadBin(i) < UInt(RADIX_DIGITS))
                {
                    short_circuit = short_circuit || bins[i] == UInt(TILE_ITEMS);
                };
            }

            short_circuit = warp_active_bit_or(short_circuit);
            $if(!short_circuit)
            {
                return;
            };
        }


        void LoadBinsToOffsetsGlobal(const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                UInt bin = ThreadBin(i);
                $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                {
                    m_global_offsets->write(bin, d_bins_in.read(bin) - offsets[i]);
                };
            }
        }

        void LoadbackGlobal(const ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                UInt bin = ThreadBin(i);
                $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                {
                    UInt inc_sum   = bins[i];
                    Int  want_mask = ~0;

                    UInt block_jdx = block_idx - 1;
                    $while(block_jdx >= 0)
                    {
                        UInt value_j = 0;

                        $while(value_j == 0)
                        {
                            value_j = d_lookback.read(block_jdx * UInt(RADIX_DIGITS) + bin);
                        };

                        inc_sum += value_j & LOOKBACK_VALUE_MASK;
                        want_mask = warp_active_bit_mask((value_j & LOOKBACK_KIND_MASK) == 0).x;
                        $if(value_j & LOOKBACK_GLOBAL_MASK)
                        {
                            $break;
                        };
                        block_jdx -= 1;
                    };
                    UInt loc_i   = d_lookback.read(block_idx * UInt(RADIX_DIGITS) + bin);
                    UInt value_i = loc_i & LOOKBACK_GLOBAL_MASK;
                    d_lookback.volatile_write(block_idx * UInt(RADIX_DIGITS) + bin, value_i);

                    m_global_offsets->write(bin, m_global_offsets->read(bin) + inc_sum - bins[i]);
                };
            }
        }

        void UpdateBinsGlobal(const ArrayVar<uint, BINS_PER_THREAD>& bins,
                              const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            Bool last_block = (block_idx + 1) * UInt(TILE_ITEMS) >= num_items;

            $if(last_block)
            {
                for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                {
                    UInt bin = ThreadBin(i);
                    $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                    {
                        d_bins_out.write(bin, m_global_offsets->read(bin) + offsets[i] + bins[i]);
                    };
                }
            };
        }


        void ShortCircuitCopy(const ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys,
                              const ArrayVar<uint, BINS_PER_THREAD>&              bins)
        {
            UInt common_bits = digit_extractor_t().Digit(keys[0]);

            ArrayVar<uint, BINS_PER_THREAD> offsets;

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                UInt bin   = ThreadBin(i);
                offsets[i] = select(0u, UInt(TILE_ITEMS), bin > common_bits);
            }

            LoadBinsToOffsetsGlobal(offsets);
            LoadbackGlobal(bins);
            UpdateBinsGlobal(bins, offsets);

            sync_block();

            UInt global_offset = m_global_offsets->read(common_bits);

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                keys[i] = Twiddle::Out(keys[i]);
            }

            $if(full_block)
            {
                StoreDirectWarpStriped(thread_id().x, d_keys_out, global_offset, keys);
            }
            $else
            {
                UInt tile_items = num_items - block_idx * UInt(TILE_ITEMS);
                StoreDirectWarpStriped(thread_id().x, d_keys_out + global_offset, keys, tile_items);
            };

            if constexpr(!KEYS_ONLY)
            {


                ArrayVar<ValueType, ITEMS_PER_THREAD> values;
                LoadValues(block_idx * UInt(TILE_ITEMS), values);
                $if(full_block)
                {
                    StoreDirectWarpStriped(thread_id().x, d_values_out + global_offset, values);
                }
                $else
                {
                    UInt tile_items = num_items - block_idx * UInt(TILE_ITEMS);
                    StoreDirectWarpStriped(thread_id().x, d_values_out + global_offset, values, tile_items);
                };
            }
        }
        void ScatterKeysShared(const ArrayVar<KeyType, ITEMS_PER_THREAD>& keys,
                               const ArrayVar<uint, ITEMS_PER_THREAD>&    ranks)
        {
            // write to shared memory
            for(int u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                m_shared_keys->write(ranks[u], keys[u]);
            }
        }


        void ScatterKeysGlobal()
        {
            UInt tile_items = select(num_items - block_idx * UInt(TILE_ITEMS), UInt(TILE_ITEMS), full_block);

            for(auto u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                UInt                  idx = thread_id().x + u * UInt(BLOCK_SIZE);
                Var<bit_ordered_type> key = m_shared_keys->read(idx);
                UInt global_idx = idx + m_global_offsets->read(digit_extractor_t().Digit(key));
                $if(FULL_BINS | idx < tile_items)
                {
                    d_keys_out.write(global_idx, Twiddle::Out(key));
                };
                sync_block();
            }
        }

        void ScatterValuesShared(const ArrayVar<ValueType, ITEMS_PER_THREAD>& values,
                                 const ArrayVar<uint, ITEMS_PER_THREAD>&      ranks)
        {
            // write to shared memory
            for(int u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                m_shared_values->write(ranks[u], values[u]);
            }
        }


        void ScatterValuesGlobal()
        {
            UInt tile_items = select(num_items - block_idx * UInt(TILE_ITEMS), UInt(TILE_ITEMS), full_block);

            for(auto u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                UInt           idx        = thread_id().x + u * UInt(BLOCK_SIZE);
                Var<ValueType> value      = m_shared_values->read(idx);
                UInt           global_idx = idx + m_global_offsets->read(value);
                $if(FULL_BINS | idx < tile_items)
                {
                    d_values_out.write(global_idx, value);
                };
                sync_block();
            }
        }


      private:
        SmemTypePtr<bit_ordered_type> m_shared_keys;
        SmemTypePtr<ValueType>        m_shared_values;
        SmemTypePtr<uint>             m_global_offsets;
        SmemTypePtr<uint>             m_block_idx;

        BufferVar<uint>& d_lookback;
        BufferVar<uint>& d_ctrs;
        BufferVar<uint>& d_bins_out;
        BufferVar<uint>& d_bins_in;

        BufferVar<KeyType>&   d_keys_in;
        BufferVar<KeyType>&   d_keys_out;
        BufferVar<ValueType>& d_values_in;
        BufferVar<ValueType>& d_values_out;

        UInt num_items;
        UInt begin_bit, end_bit;

        UInt current_bit;
        UInt warp;
        UInt lane_id;
        UInt block_idx;
        Bool full_block;
    };
};  // namespace details
}  // namespace luisa::parallel_primitive