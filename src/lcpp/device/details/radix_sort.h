/*
 * @Author: Ligo 
 * @Date: 2025-11-12 14:58:11 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 00:20:27
 */


#pragma once
#include "luisa/dsl/stmt.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <lcpp/agent/agent_reduce.h>
#include <lcpp/agent/agent_radix_sort_histogram.h>
#include <lcpp/agent/agenet_radix_sort_onesweep.h>
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
    template <NumericT KeyType, bool IS_DESCENDING, size_t RADIX_BIT = 8u, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class RadixSortHistogramModule : public LuisaModule
    {
      public:
        using RadixSortHistogramKernel = Shader<1, Buffer<uint>, Buffer<KeyType>, uint, uint, uint>;

        U<RadixSortHistogramKernel> compile(Device& device)
        {
            U<RadixSortHistogramKernel> ms_radix_sort_histogram_shader = nullptr;

            lazy_compile(
                device,
                ms_radix_sort_histogram_shader,
                [&](BufferVar<uint> d_bins_out, BufferVar<KeyType> d_keys_in, UInt num_elements, UInt start_bit, UInt end_bit) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    set_warp_size(WARP_SIZE);
                    using HistogramPolicy =
                        AgentRadixSortHistogramPolicy<BLOCK_SIZE, ITEMS_PER_THREAD, 1u, KeyType, RADIX_BIT>;
                    using AgentT =
                        AgentRadixSortHistogram<KeyType, IS_DESCENDING, HistogramPolicy::RADIX_BITS, HistogramPolicy::NUM_PARTS, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;

                    SmemTypePtr<uint> s_bins = new SmemType<uint>{AgentT::SHARED_MEM_SIZE};

                    AgentT agent(s_bins, d_bins_out, d_keys_in, num_elements, start_bit, end_bit);
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
        using RadixSortExclusiveSumKernel = Shader<1, Buffer<uint>, Buffer<uint>>;

        U<RadixSortExclusiveSumKernel> compile(Device& device)
        {
            U<RadixSortExclusiveSumKernel> ms_radix_sort_exclusive_sum_shader = nullptr;
            lazy_compile(device,
                         ms_radix_sort_exclusive_sum_shader,
                         [&](BufferVar<uint> d_bins_in, BufferVar<uint> d_bins_out)
                         {
                             constexpr uint BINS_PER_THREAD = (RADIX_DIGIT + BLOCK_SIZE - 1) / BLOCK_SIZE;

                             set_block_size(BLOCK_SIZE);
                             set_warp_size(WARP_SIZE);
                             ArrayVar<uint, BINS_PER_THREAD> bins;

                             UInt bin_start = block_id().x * UInt(RADIX_DIGIT);
                             for(auto i = 0u; i < BINS_PER_THREAD; ++i)
                             {
                                 UInt bin_index = thread_id().x * BINS_PER_THREAD + i;
                                 $if(bin_index < UInt(RADIX_DIGIT))
                                 {
                                     bins[i] = d_bins_in.read(bin_start + bin_index);
                                 };
                             }

                             ArrayVar<uint, BINS_PER_THREAD> bins_out;
                             BlockScan<uint, BLOCK_SIZE, BINS_PER_THREAD, WARP_SIZE>().ExclusiveSum(bins, bins_out);
                             for(auto i = 0u; i < BINS_PER_THREAD; ++i)
                             {
                                 UInt bin_index = thread_id().x * BINS_PER_THREAD + i;
                                 $if(bin_index < UInt(RADIX_DIGIT))
                                 {
                                     d_bins_out.write(bin_start + bin_index, bins_out[i]);
                                 };
                             }
                         });
            return ms_radix_sort_exclusive_sum_shader;
        };
    };


    using namespace luisa::compute;
    template <NumericT KeyType, NumericT ValueType, bool IS_DESCENDING, size_t RADIX_BIT = 8u, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class RadixSortOneSweepModule : public LuisaModule
    {
      public:
        using RadixSortOneSweepKernel = Shader<1, Buffer<uint>, Buffer<uint>>;

        U<RadixSortOneSweepKernel> compile(Device& device)
        {
            U<RadixSortOneSweepKernel> ms_radix_sort_exclusive_sum_shader = nullptr;
            lazy_compile(
                device,
                ms_radix_sort_exclusive_sum_shader,
                [&](BufferVar<uint> d_bins_in, BufferVar<uint> d_bins_out)
                {
                    set_block_size(BLOCK_SIZE);
                    set_warp_size(WARP_SIZE);

                    set_block_size(BLOCK_SIZE);
                    set_warp_size(WARP_SIZE);
                    using RadixSortOneSweepPolicy = AgentRadixSortOneSweepPolicy<1u, 8u, KeyType, 1u, RADIX_BIT>;
                    using AgentT =
                        AgentRadixSortOneSweep<KeyType, ValueType, RADIX_BIT, RadixSortOneSweepPolicy::RANK_NUM_PARTS, false, IS_DESCENDING, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;

                    // SmemTypePtr<uint> s_bins = new SmemType<uint>{AgentT::SHARED_MEM_SIZE};

                    // AgentT agent(s_bins, d_bins_out, d_keys_in, num_elements, start_bit, end_bit);
                    // agent.Process();
                });
            return ms_radix_sort_exclusive_sum_shader;
        };
    };
}  // namespace details
}  // namespace luisa::parallel_primitive