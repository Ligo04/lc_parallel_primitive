/*
 * @Author: Ligo 
 * @Date: 2025-11-12 14:58:11 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 09:59:23
 */


#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/runtime/byte_buffer.h>
#include <lcpp/agent/agent_reduce.h>
#include <lcpp/agent/agent_radix_sort_histogram.h>
#include <lcpp/agent/agent_radix_sort_onesweep.h>
#include <lcpp/agent/policy.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_scan.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <NumericT KeyType>
    class RadixSortResetModule : public LuisaModule
    {
      public:
        using RadixSortResetKernel = Shader<1, Buffer<KeyType>, KeyType>;

        U<RadixSortResetKernel> compile(Device& device)
        {
            U<RadixSortResetKernel> ms_radix_sort_reset_shader = nullptr;

            lazy_compile(device,
                         ms_radix_sort_reset_shader,
                         [&](BufferVar<KeyType> d_bins_out, Var<KeyType> init_value) noexcept
                         { d_bins_out.write(compute::dispatch_id().x, init_value); });
            return ms_radix_sort_reset_shader;
        };
    };

    using namespace luisa::compute;
    template <NumericT KeyType, bool IS_DESCENDING, size_t RADIX_BIT = 8u, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class RadixSortHistogramModule : public LuisaModule
    {
      public:
        using RadixSortHistogramKernel = Shader<1, Buffer<uint>, ByteBuffer, uint, uint, uint>;

        U<RadixSortHistogramKernel> compile(Device& device)
        {
            U<RadixSortHistogramKernel> ms_radix_sort_histogram_shader = nullptr;

            lazy_compile(
                device,
                ms_radix_sort_histogram_shader,
                [&](BufferVar<uint> d_bins_out, const ByteBufferVar& d_keys_in, UInt num_elements, UInt start_bit, UInt end_bit) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    set_warp_size(WARP_SIZE);
                    using HistogramPolicy =
                        AgentRadixSortHistogramPolicy<BLOCK_SIZE, ITEMS_PER_THREAD, 1u, KeyType, RADIX_BIT>;
                    using AgentT =
                        AgentRadixSortHistogram<KeyType, IS_DESCENDING, HistogramPolicy::RADIX_BITS, HistogramPolicy::NUM_PARTS, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;

                    AgentT agent(d_bins_out, d_keys_in, num_elements, start_bit, end_bit);
                    agent.Process();
                });
            return ms_radix_sort_histogram_shader;
        };
    };

    using namespace luisa::compute;
    template <size_t RADIX_DIGIT, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE>
    class RadixSortExclusiveSumModule : public LuisaModule
    {
      public:
        using RadixSortExclusiveSumKernel = Shader<1, Buffer<uint>>;

        U<RadixSortExclusiveSumKernel> compile(Device& device)
        {
            U<RadixSortExclusiveSumKernel> ms_radix_sort_exclusive_sum_shader = nullptr;
            lazy_compile(device,
                         ms_radix_sort_exclusive_sum_shader,
                         [&](BufferVar<uint> d_bins)
                         {
                             set_block_size(BLOCK_SIZE);
                             set_warp_size(WARP_SIZE);

                             constexpr uint BINS_PER_THREAD = (RADIX_DIGIT + BLOCK_SIZE - 1) / BLOCK_SIZE;
                             ArrayVar<uint, BINS_PER_THREAD> bins;

                             UInt bin_start = block_id().x * UInt(RADIX_DIGIT);
                             for(auto i = 0u; i < BINS_PER_THREAD; ++i)
                             {
                                 UInt bin_index = thread_id().x * BINS_PER_THREAD + i;
                                 $if(bin_index < UInt(RADIX_DIGIT))
                                 {
                                     bins[i] = d_bins.read(bin_start + bin_index);
                                 };
                             }

                             BlockScan<uint, BLOCK_SIZE, BINS_PER_THREAD, WARP_SIZE>().ExclusiveSum(bins, bins);
                             for(auto i = 0u; i < BINS_PER_THREAD; ++i)
                             {
                                 UInt bin_index = thread_id().x * BINS_PER_THREAD + i;
                                 $if(bin_index < UInt(RADIX_DIGIT))
                                 {
                                     d_bins.write(bin_start + bin_index, bins[i]);
                                 };
                             }
                         });
            return ms_radix_sort_exclusive_sum_shader;
        };
    };


    template <NumericT KeyType, typename ValueType, bool KEY_ONLY, bool IS_DESCENDING, size_t RADIX_BIT = 8u, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class RadixSortOneSweepModule : public LuisaModule
    {
      public:
        // key value pair
        using RadixSortOneSweepKernel =
            Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, ByteBuffer, ByteBuffer, Buffer<ValueType>, Buffer<ValueType>, uint, uint, uint>;

        U<RadixSortOneSweepKernel> compile(Device& device)
        {
            U<RadixSortOneSweepKernel> ms_radix_sort_onesweep_kernel = nullptr;
            lazy_compile(
                device,
                ms_radix_sort_onesweep_kernel,
                [&](BufferVar<uint>      d_lookback,
                    BufferVar<uint>      d_ctrs,
                    BufferVar<uint>      d_bins_in,
                    BufferVar<uint>      d_bins_out,
                    ByteBufferVar        d_keys_in,
                    ByteBufferVar        d_keys_out,
                    BufferVar<ValueType> d_values_in,
                    BufferVar<ValueType> d_values_out,
                    compute::UInt        num_items,
                    compute::UInt        current_bit,
                    compute::UInt        num_bits) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    set_warp_size(WARP_SIZE);

                    using RadixSortOneSweepPolicy = AgentRadixSortOneSweepPolicy<1u, 8u, KeyType, 1u, RADIX_BIT>;
                    using AgentT =
                        AgentRadixSortOneSweep<KeyType, ValueType, KEY_ONLY, RADIX_BIT, RadixSortOneSweepPolicy::RANK_NUM_PARTS, IS_DESCENDING, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;

                    AgentT agent(
                        d_lookback, d_ctrs, d_bins_in, d_bins_out, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, current_bit, num_bits);
                    agent.Process();
                });
            return ms_radix_sort_onesweep_kernel;
        };
    };
}  // namespace details
}  // namespace luisa::parallel_primitive