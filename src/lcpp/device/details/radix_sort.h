/*
 * @Author: Ligo 
 * @Date: 2025-11-12 14:58:11 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 00:20:27
 */


#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <lcpp/agent/agent_reduce.h>
#include <lcpp/agent/agent_radix_sort_histogram.h>
#include <lcpp/agent/policy.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <NumericT KeyType, bool IS_DESCENDING, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
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
                        AgentRadixSortHistogramPolicy<BLOCK_SIZE, ITEMS_PER_THREAD, 1u, KeyType, 8u>;
                    using AgentT =
                        AgentRadixSortHistogram<KeyType, IS_DESCENDING, HistogramPolicy::RADIX_BITS, HistogramPolicy::NUM_PARTS, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;

                    SmemTypePtr<uint> s_bins = new SmemType<uint>{AgentT::SHARED_MEM_SIZE};

                    AgentT agent(s_bins, d_bins_out, d_keys_in, num_elements, start_bit, end_bit);
                    agent.Process();
                });
            return ms_radix_sort_histogram_shader;
        };
    };
}  // namespace details
}  // namespace luisa::parallel_primitive