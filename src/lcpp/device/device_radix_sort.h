/*
 * @Author: Ligo 
 * @Date: 2025-11-12 11:08:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 00:15:11
 */


#pragma once

#include <algorithm>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <limits>
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
    bool   m_created = false;

  public:
    DeviceRadixSort()  = default;
    ~DeviceRadixSort() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    template <NumericT KeyType, NumericT ValueType>
    void SortPairs(CommandList&          cmdlist,
                   Stream&               stream,
                   BufferView<KeyType>   d_keys_in,
                   BufferView<ValueType> d_values_in,
                   BufferView<KeyType>   d_keys_out,
                   BufferView<ValueType> d_values_out,
                   uint                  num_items)
    {
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        onesweep_radix_sort<KeyType, ValueType, false>(
            cmdlist, stream, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, true);
    };

    template <NumericT KeyType>
    void SortKeys(CommandList& cmdlist, Stream& stream, BufferView<KeyType> d_keys_in, BufferView<KeyType> d_keys_out, uint num_items)
    {
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        onesweep_radix_sort<KeyType, false>(cmdlist, stream, d_keys, 0, sizeof(KeyType) * 8, num_items, true);
    };

  private:
    template <NumericT KeyType, bool is_descending>
    void onesweep_radix_sort(CommandList&           cmdlist,
                             Stream&                stream,
                             DoubleBuffer<KeyType>& d_keys,
                             uint                   begin_bit,
                             uint                   end_bit,
                             uint                   num_items,
                             bool                   is_overwrite_okay)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RAIDX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        const uint ONESWEEP_ITMES_PER_THREADS = ITEMS_PER_THREAD;
        const uint ONESWEEP_BLOCK_THREADS     = m_block_size;
        const uint ONESWEEP_TILE_ITEMS        = ONESWEEP_ITMES_PER_THREADS * ONESWEEP_BLOCK_THREADS;

        const auto PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div(end_bit - begin_bit, RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t value_size         = 0;
        size_t allocation_sizes[] = {
            // bins
            num_portions * num_passes * RADIX_DIGITS * sizeof(uint),
            // lookback
            max_num_blocks * RADIX_DIGITS * sizeof(uint),
            // extra key buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items * sizeof(KeyType),
            // extra value buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items * value_size,
            // counters
            num_portions * num_passes * sizeof(uint),
        };

        auto d_bins_buffer = m_device.create_buffer<uint>(allocation_sizes[0]);

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, is_descending, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using RadixSortHistogramKernel = RadixSortHistogram::RadixSortHistogramKernel;
        auto radix_sort_histogram_key = luisa::string(luisa::compute::Type::of<KeyType>()->description())
                                        + luisa::string(is_descending ? "_desc" : "_asc");
        auto ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_histogram_key);
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            auto shader = RadixSortHistogram().compile(m_device);
            ms_radix_sort_histogram_map.try_emplace(radix_sort_histogram_key, std::move(shader));
            ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_histogram_key);
        }
        auto ms_radix_sort_histogram_ptr =
            reinterpret_cast<RadixSortHistogramKernel*>(&(*ms_radix_sort_histogram_it->second));
        const auto num_sms             = 128;
        const auto histo_blocks_per_sm = 2;
        LUISA_INFO("max_num_blocks * num_portions: {} * {} = {},num_items:{}",
                   max_num_blocks,
                   num_portions,
                   max_num_blocks * num_portions,
                   num_items);
        LUISA_INFO("bins size: num_portions:{}, num_passes: {}, RADIX_DIGITS: {}", num_portions, num_passes, RADIX_DIGITS);
        cmdlist << (*ms_radix_sort_histogram_ptr)(d_bins_buffer.view(), d_keys.current(), num_items, begin_bit, end_bit)
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
        auto d_bins_sum_buffer = m_device.create_buffer<uint>(allocation_sizes[0]);

        using RadixSortExclusiveSum = details::RadixSortExclusiveSumModule<RADIX_DIGITS, BLOCK_SIZE, WARP_NUMS>;
        using RadixSortExclusiveSumKernel = RadixSortExclusiveSum::RadixSortExclusiveSumKernel;
        auto radix_sort_exclusive_sum_key = luisa::string(luisa::compute::Type::of<uint>()->description())
                                            + luisa::string(is_descending ? "_desc" : "_asc");
        auto ms_radix_sort_exclusive_sum_it = ms_radix_sort_exclusive_sum_map.find(radix_sort_exclusive_sum_key);
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end())
        {
            auto shader = RadixSortExclusiveSum().compile(m_device);
            ms_radix_sort_exclusive_sum_map.try_emplace(radix_sort_exclusive_sum_key, std::move(shader));
            ms_radix_sort_exclusive_sum_it = ms_radix_sort_exclusive_sum_map.find(radix_sort_exclusive_sum_key);
        }
        auto ms_radix_sort_exclusive_sum_ptr =
            reinterpret_cast<RadixSortExclusiveSumKernel*>(&(*ms_radix_sort_exclusive_sum_it->second));
        cmdlist
            << (*ms_radix_sort_exclusive_sum_ptr)(d_bins_buffer.view(), d_bins_sum_buffer.view()).dispatch(num_passes * m_block_size);
        stream << cmdlist.commit() << synchronize();

        //show
        // luisa::vector<uint> host_bins(d_bins_sum_buffer.size());
        // stream << d_bins_sum_buffer.copy_to(host_bins.data()) << synchronize();
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
        auto d_key_tmp = d_keys.alternate();
        // auto d_value_tmp = d_values.alternate();
        if(!is_overwrite_okay && num_passes % 2 == 0)
        {
            d_keys.d_buffer[1] = d_key_tmp;
        }

        Buffer<uint> d_lookback_buffer;

        for(uint current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            uint num_bit = std::min(end_bit - current_bit, RADIX_BITS);

            for(uint portion = 0; portion < num_portions; ++portion)
            {
                uint portion_num_items = std::min(num_items - portion * PORTION_SIZE, PORTION_SIZE);

                uint num_blocks = ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

                d_lookback_buffer = m_device.create_buffer<uint>(num_blocks * RADIX_DIGITS);
            }

            if(!is_overwrite_okay && pass == 0)
            {
                // d_keys   = num_passes % 2 == 0 ? DoubleBuffer<KeyT>(d_keys_tmp, d_keys_tmp2) :
                //                                  DoubleBuffer<KeyT>(d_keys_tmp2, d_keys_tmp);
                // d_values = num_passes % 2 == 0 ? DoubleBuffer<ValueT>(d_values_tmp, d_values_tmp2) :
                //                                  DoubleBuffer<ValueT>(d_values_tmp2, d_values_tmp);
            }
            d_keys.selector ^= 1;
            // d_values.selector ^= 1;
        }
    }


    template <NumericT KeyType, NumericT ValueType, bool is_descending>
    void onesweep_radix_sort(CommandList&             cmdlist,
                             Stream&                  stream,
                             DoubleBuffer<KeyType>&   d_keys,
                             DoubleBuffer<ValueType>& d_values,
                             uint                     begin_bit,
                             uint                     end_bit,
                             uint                     num_items,
                             bool                     is_overwrite_okay)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RAIDX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        const uint ONESWEEP_ITMES_PER_THREADS = ITEMS_PER_THREAD;
        const uint ONESWEEP_BLOCK_THREADS     = m_block_size;
        const uint ONESWEEP_TILE_ITEMS        = ONESWEEP_ITMES_PER_THREADS * ONESWEEP_BLOCK_THREADS;

        const auto PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div(end_bit - begin_bit, RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t value_size         = sizeof(ValueType);
        size_t allocation_sizes[] = {
            // bins
            num_portions * num_passes * RADIX_DIGITS,
            // lookback
            max_num_blocks * RADIX_DIGITS,
            // extra key buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items,
            // extra value buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items,
            // counters
            num_portions * num_passes,
        };

        auto d_bin_in_buffer = m_device.create_buffer<uint>(allocation_sizes[0]);

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, is_descending, ONESWEEP_BLOCK_THREADS, WARP_NUMS, ONESWEEP_ITMES_PER_THREADS>;
        using RadixSortHistogramKernel = RadixSortHistogram::RadixSortHistogramKernel;
        auto radix_sort_histogram_key  = get_type_and_op_desc<KeyType, ValueType>()
                                        + luisa::string(is_descending ? "_desc" : "_asc");
        auto ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_histogram_key);
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            auto shader = RadixSortHistogram().compile(m_device);
            ms_radix_sort_histogram_map.try_emplace(radix_sort_histogram_key, std::move(shader));
            ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_histogram_key);
        }
        auto ms_radix_sort_histogram_ptr =
            reinterpret_cast<RadixSortHistogramKernel*>(&(*ms_radix_sort_histogram_it->second));

        cmdlist << (*ms_radix_sort_histogram_ptr)(d_bin_in_buffer.view(), d_keys.current(), num_items, begin_bit, end_bit)
                       .dispatch(max_num_blocks * num_portions);
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_histogram_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_exclusive_sum_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_onesweep_radix_sort_map;
};
}  // namespace luisa::parallel_primitive