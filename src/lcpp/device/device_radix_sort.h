/*
 * @Author: Ligo 
 * @Date: 2025-11-12 11:08:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 15:50:13
 */


#pragma once

#include <algorithm>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/utils.h>
#include <lcpp/agent/policy.h>
#include <lcpp/device/details/radix_sort.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceRadixSort : public LuisaModule
{
    enum class RadixSortAlgorithm
    {
        ONE_SWEEP = 0,
    };

  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;

  public:
    DeviceRadixSort()  = default;
    ~DeviceRadixSort() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
    }

    template <NumericT KeyType, NumericT ValueType>
    void SortPairs(CommandList&          cmdlist,
                   Stream&               stream,
                   BufferView<KeyType>   d_keys_in,
                   BufferView<KeyType>   d_keys_out,
                   BufferView<ValueType> d_values_in,
                   BufferView<ValueType> d_values_out,
                   uint                  num_items)
    {
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        onesweep_radix_sort<KeyType, ValueType, false, false>(
            cmdlist, stream, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false);
    };

    template <NumericT KeyType>
    void SortKeys(CommandList& cmdlist, Stream& stream, BufferView<KeyType> d_keys_in, BufferView<KeyType> d_keys_out, uint num_items)
    {
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        onesweep_radix_sort<KeyType, KeyType, true, false>(
            cmdlist, stream, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false);
    };


    template <NumericT KeyType, NumericT ValueType>
    void SortPairsDescending(CommandList&          cmdlist,
                             Stream&               stream,
                             BufferView<KeyType>   d_keys_in,
                             BufferView<KeyType>   d_keys_out,
                             BufferView<ValueType> d_values_in,
                             BufferView<ValueType> d_values_out,
                             uint                  num_items)
    {
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        onesweep_radix_sort<KeyType, ValueType, false, true>(
            cmdlist, stream, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false);
    };

    template <NumericT KeyType>
    void SortKeysDescending(CommandList&        cmdlist,
                            Stream&             stream,
                            BufferView<KeyType> d_keys_in,
                            BufferView<KeyType> d_keys_out,
                            uint                num_items)
    {
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        onesweep_radix_sort<KeyType, KeyType, true, true>(
            cmdlist, stream, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false);
    };

  private:
    template <NumericT KeyType, typename ValueType, bool KEY_ONLY, bool IS_DESCENDING>
    void onesweep_radix_sort(CommandList&             cmdlist,
                             Stream&                  stream,
                             DoubleBuffer<KeyType>&   d_keys,
                             DoubleBuffer<ValueType>& d_values,
                             uint                     begin_bit,
                             uint                     end_bit,
                             uint                     num_items,
                             bool                     is_overwrite_okay)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        const uint ONESWEEP_ITMES_PER_THREADS = ITEMS_PER_THREAD;
        const uint ONESWEEP_BLOCK_THREADS     = m_block_size;
        const uint ONESWEEP_TILE_ITEMS        = ONESWEEP_ITMES_PER_THREADS * ONESWEEP_BLOCK_THREADS;

        const auto PORTION_SIZE = ((1u << 28u) - 1u) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div(end_bit - begin_bit, RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t value_size         = 0;
        size_t allocation_sizes[] = {
            // bins
            num_portions * num_passes * RADIX_DIGITS,
            // lookback
            max_num_blocks * RADIX_DIGITS,
            // extra key buffer
            num_items,
            // counters
            num_portions * num_passes,
        };

        auto              d_bins_buffer     = m_device.create_buffer<uint>(allocation_sizes[0]);
        auto              d_lookback_buffer = m_device.create_buffer<uint>(allocation_sizes[1]);
        Buffer<KeyType>   d_keys_tmp2_buffer;
        Buffer<ValueType> d_values_tmp2_buffer;
        if(!is_overwrite_okay && num_passes > 1)
        {
            d_keys_tmp2_buffer   = m_device.create_buffer<KeyType>(allocation_sizes[2]);
            d_values_tmp2_buffer = m_device.create_buffer<ValueType>(allocation_sizes[2]);
        }
        auto d_ctrs_buffer = m_device.create_buffer<uint>(allocation_sizes[3]);

        auto radix_sort_key = get_type_and_op_desc<KeyType, ValueType>()
                              + luisa::string(IS_DESCENDING ? "_desc" : "_asc");


        // reset keys
        using RadixSortReset        = details::RadixSortResetModule<uint>;
        using RadixSortResetKernel  = RadixSortReset::RadixSortResetKernel;
        auto ms_radix_sort_reset_it = ms_radix_sort_reset_map.find(radix_sort_key);
        if(ms_radix_sort_reset_it == ms_radix_sort_reset_map.end())
        {
            auto shader = RadixSortReset().compile(m_device);
            auto [it, inserted] = ms_radix_sort_reset_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_reset_it = it;
        }
        auto ms_radix_sort_reset_ptr =
            reinterpret_cast<RadixSortResetKernel*>(&(*ms_radix_sort_reset_it->second));

        cmdlist << (*ms_radix_sort_reset_ptr)(d_bins_buffer.view(), 0u).dispatch(d_bins_buffer.size())
                << (*ms_radix_sort_reset_ptr)(d_ctrs_buffer.view(), 0u).dispatch(d_ctrs_buffer.size());

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using RadixSortHistogramKernel  = RadixSortHistogram::RadixSortHistogramKernel;
        auto ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_key);
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            auto shader = RadixSortHistogram().compile(m_device);
            auto [it, inserted] = ms_radix_sort_histogram_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_histogram_it = it;
        }
        auto ms_radix_sort_histogram_ptr =
            reinterpret_cast<RadixSortHistogramKernel*>(&(*ms_radix_sort_histogram_it->second));
        const auto num_sms             = BLOCK_SIZE;
        const auto histo_blocks_per_sm = 1;
        // LUISA_INFO("max_num_blocks * num_portions: {} * {} = {},num_items:{}",
        //            max_num_blocks,
        //            num_portions,
        //            max_num_blocks * num_portions,
        //            num_items);
        // LUISA_INFO("bins size: num_portions:{}, num_passes: {}, RADIX_DIGITS: {}", num_portions, num_passes, RADIX_DIGITS);
        cmdlist << (*ms_radix_sort_histogram_ptr)(
                       d_bins_buffer.view(), ByteBufferView{d_keys.current()}, num_items, begin_bit, end_bit)
                       .dispatch(num_sms * histo_blocks_per_sm * m_block_size);
        stream << cmdlist.commit() << synchronize();

        // luisa::vector<uint> host_bins(d_bins_buffer.size());
        // stream << d_bins_buffer.copy_to(host_bins.data()) << synchronize();
        // for(auto i = 0; i < num_passes; ++i)
        // {
        //     LUISA_INFO("Pass {}", i);
        //     for(auto j = 0; j < RADIX_DIGITS; ++j)
        //     {
        //         uint sum = 0;
        //         for(auto p = 0; p < num_portions; ++p)
        //         {
        //             auto index = i * RADIX_DIGITS * num_portions + j * num_portions + p;
        //             sum += host_bins[index];
        //         }
        //         LUISA_INFO("  Bin {}: {}", j, sum);
        //     }
        // }


        // exclusive scan
        using RadixSortExclusiveSum = details::RadixSortExclusiveSumModule<RADIX_DIGITS, BLOCK_SIZE, WARP_NUMS>;
        using RadixSortExclusiveSumKernel   = RadixSortExclusiveSum::RadixSortExclusiveSumKernel;
        auto ms_radix_sort_exclusive_sum_it = ms_radix_sort_exclusive_sum_map.find(radix_sort_key);
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end())
        {
            auto shader = RadixSortExclusiveSum().compile(m_device);
            auto [it, inserted] =
                ms_radix_sort_exclusive_sum_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_exclusive_sum_it = it;
        }
        auto ms_radix_sort_exclusive_sum_ptr =
            reinterpret_cast<RadixSortExclusiveSumKernel*>(&(*ms_radix_sort_exclusive_sum_it->second));

        cmdlist << (*ms_radix_sort_exclusive_sum_ptr)(d_bins_buffer.view()).dispatch(num_passes * m_block_size);
        stream << cmdlist.commit() << synchronize();

        //show
        // stream << d_bins_buffer.copy_to(host_bins.data()) << synchronize();
        // for(auto i = 0; i < num_passes; ++i)
        // {
        //     LUISA_INFO("Pass {}", i);
        //     for(auto j = 0; j < RADIX_DIGITS; ++j)
        //     {
        //         uint sum = 0;
        //         for(auto p = 0; p < num_portions; ++p)
        //         {
        //             auto index = i * RADIX_DIGITS * num_portions + j * num_portions + p;
        //             sum += host_bins[index];
        //         }
        //         LUISA_INFO("  Bin {}: {}", j, sum);
        //     }
        // }

        // one sweep
        auto d_keys_tmp   = d_keys.alternate();
        auto d_values_tmp = d_values.alternate();
        if(!is_overwrite_okay && num_passes % 2 == 0)
        {
            d_keys.d_buffer[1]   = d_keys_tmp2_buffer.view();
            d_values.d_buffer[1] = d_values_tmp2_buffer.view();
        }

        using RadixSortOneSweep =
            details::RadixSortOneSweepModule<KeyType, ValueType, KEY_ONLY, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ONESWEEP_ITMES_PER_THREADS>;
        using RadixSortOneSweepKernel = RadixSortOneSweep::RadixSortOneSweepKernel;

        auto ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(radix_sort_key);
        if(ms_radix_sort_onesweep_it == ms_radix_sort_one_sweep_map.end())
        {
            auto shader = RadixSortOneSweep().compile(m_device);
            ms_radix_sort_one_sweep_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(radix_sort_key);
        }
        auto ms_radix_sort_onesweep_ptr =
            reinterpret_cast<RadixSortOneSweepKernel*>(&(*ms_radix_sort_onesweep_it->second));

        for(uint current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            uint num_bit = std::min(end_bit - current_bit, RADIX_BITS);

            for(uint portion = 0; portion < num_portions; ++portion)
            {
                uint portion_num_items = std::min(num_items - portion * PORTION_SIZE, PORTION_SIZE);
                uint num_blocks        = ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

                // LUISA_INFO("  Pass {}, Portion {}, portion_num_items: {}, num_blocks: {}", pass, portion, portion_num_items, num_blocks);

                cmdlist << (*ms_radix_sort_reset_ptr)(d_lookback_buffer.view(), 0u)
                               .dispatch(d_lookback_buffer.size());

                // dispatch
                cmdlist
                    << (*ms_radix_sort_onesweep_ptr)(
                           d_lookback_buffer.view(),
                           d_ctrs_buffer.view(portion * num_passes + pass, 1),
                           d_bins_buffer.view((portion * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS),
                           portion < num_portions - 1 ?
                               d_bins_buffer.view(((portion + 1) * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS) :
                               d_bins_buffer.view(0, 0),
                           ByteBufferView{d_keys.current().subview(portion * PORTION_SIZE, portion_num_items)},
                           ByteBufferView{d_keys.alternate()},
                           KEY_ONLY ? d_values.current().subview(0, 0) :
                                      d_values.current().subview(portion * PORTION_SIZE, portion_num_items),
                           KEY_ONLY ? d_values.alternate().subview(0, 0) :
                                      d_values.alternate().subview(portion * PORTION_SIZE, portion_num_items),
                           portion_num_items,
                           current_bit,
                           num_bit)
                           .dispatch(num_blocks * ONESWEEP_BLOCK_THREADS);
            }
            if(!is_overwrite_okay && pass == 0)
            {
                d_keys   = num_passes % 2 == 0 ?
                               DoubleBuffer<KeyType>(d_keys_tmp, d_keys_tmp2_buffer.view()) :
                               DoubleBuffer<KeyType>(d_keys_tmp2_buffer.view(), d_keys_tmp);
                d_values = num_passes % 2 == 0 ?
                               DoubleBuffer<ValueType>(d_values_tmp, d_values_tmp2_buffer.view()) :
                               DoubleBuffer<ValueType>(d_values_tmp2_buffer.view(), d_values_tmp);
            }
            d_keys.selector ^= 1;
            d_values.selector ^= 1;
        }

        stream << cmdlist.commit() << synchronize();

        d_bins_buffer.release();
        d_lookback_buffer.release();
        d_ctrs_buffer.release();
        if(!is_overwrite_okay && num_passes > 1)
        {
            d_keys_tmp2_buffer.release();
            d_values_tmp2_buffer.release();
        }
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_histogram_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_exclusive_sum_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_one_sweep_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_reset_map;
};
}  // namespace luisa::parallel_primitive