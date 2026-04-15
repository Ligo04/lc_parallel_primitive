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

#ifndef NDEBUG
    Stream* m_debug_stream; // bind debug stream for sync
#endif
    inline Stream* debug_stream() noexcept
    {
#ifndef NDEBUG
        return m_debug_stream;
#else
        return nullptr;
#endif
    }
    void create(Device& device, Stream* debug_stream = nullptr)
    {
        m_device                   = device;
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
    }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for SortPairs / SortPairsDescending
    template <typename KeyType, typename ValueType>
    static size_t GetSortPairsTempStorageBytes(uint num_items)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        const uint ONESWEEP_TILE_ITEMS = ITEMS_PER_THREAD * BLOCK_SIZE;
        const auto PORTION_SIZE = ((1u << 28u) - 1u) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div((uint)(sizeof(KeyType) * 8), RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t bytes = 0;
        // d_bins: num_portions * num_passes * RADIX_DIGITS * uint
        size_t bins_count = (size_t)num_portions * num_passes * RADIX_DIGITS;
        bytes += bins_count * sizeof(uint);
        // d_lookback: max_num_blocks * RADIX_DIGITS * uint
        size_t lookback_count = (size_t)max_num_blocks * RADIX_DIGITS;
        bytes += lookback_count * sizeof(uint);
        // extra key buffer (for multi-pass): num_items * sizeof(KeyType)
        if(num_passes > 1)
        {
            bytes = align_up_uint(bytes, alignof(KeyType));
            bytes += (size_t)num_items * sizeof(KeyType);
            // extra value buffer
            bytes = align_up_uint(bytes, alignof(ValueType));
            bytes += (size_t)num_items * sizeof(ValueType);
        }
        // d_ctrs: num_portions * num_passes * uint
        size_t ctrs_count = (size_t)num_portions * num_passes;
        bytes += ctrs_count * sizeof(uint);
        return bytes;
    }

    /// Temp storage bytes for SortKeys / SortKeysDescending
    template <typename KeyType>
    static size_t GetSortKeysTempStorageBytes(uint num_items)
    {
        return GetSortPairsTempStorageBytes<KeyType, KeyType>(num_items);
    }


    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT KeyType, NumericT ValueType>
    void SortPairs(CommandList&          cmdlist,
                   BufferView<uint>      temp_storage,
                   BufferView<KeyType>   d_keys_in,
                   BufferView<KeyType>   d_keys_out,
                   BufferView<ValueType> d_values_in,
                   BufferView<ValueType> d_values_out,
                   uint                  num_items)
    {
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        lcpp_check(onesweep_radix_sort<KeyType, ValueType, false, false>(
            cmdlist, temp_storage, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false),
            cmdlist, debug_stream());
    };

    template <NumericT KeyType>
    void SortKeys(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<KeyType> d_keys_in, BufferView<KeyType> d_keys_out, uint num_items)
    {
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, false>(
            cmdlist, temp_storage, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false),
            cmdlist, debug_stream());
    };


    template <NumericT KeyType, NumericT ValueType>
    void SortPairsDescending(CommandList&          cmdlist,
                             BufferView<uint>      temp_storage,
                             BufferView<KeyType>   d_keys_in,
                             BufferView<KeyType>   d_keys_out,
                             BufferView<ValueType> d_values_in,
                             BufferView<ValueType> d_values_out,
                             uint                  num_items)
    {
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        lcpp_check(onesweep_radix_sort<KeyType, ValueType, false, true>(
            cmdlist, temp_storage, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false),
            cmdlist, debug_stream());
    };

    template <NumericT KeyType>
    void SortKeysDescending(CommandList&        cmdlist,
                            BufferView<uint>    temp_storage,
                            BufferView<KeyType> d_keys_in,
                            BufferView<KeyType> d_keys_out,
                            uint                num_items)
    {
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, true>(
            cmdlist, temp_storage, d_keys, d_values, 0, sizeof(KeyType) * 8, num_items, false),
            cmdlist, debug_stream());
    };

  private:
    template <NumericT KeyType, typename ValueType, bool KEY_ONLY, bool IS_DESCENDING>
    [[nodiscard]] int onesweep_radix_sort(CommandList&             cmdlist,
                             BufferView<uint>         temp_storage,
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

        // Carve out sub-buffers from temp_storage
        size_t offset_bytes = 0;

        // d_bins: num_portions * num_passes * RADIX_DIGITS * uint
        size_t bins_count = (size_t)num_portions * num_passes * RADIX_DIGITS;
        auto d_bins_view = temp_storage.subview(offset_bytes / sizeof(uint), bins_count);
        offset_bytes += bins_count * sizeof(uint);

        // d_lookback: max_num_blocks * RADIX_DIGITS * uint
        size_t lookback_count = (size_t)max_num_blocks * RADIX_DIGITS;
        auto d_lookback_view = temp_storage.subview(offset_bytes / sizeof(uint), lookback_count);
        offset_bytes += lookback_count * sizeof(uint);

        // extra key/value buffers for multi-pass
        BufferView<KeyType>   d_keys_tmp2_view;
        BufferView<ValueType> d_values_tmp2_view;
        if(!is_overwrite_okay && num_passes > 1)
        {
            offset_bytes = align_up_uint(offset_bytes, alignof(KeyType));
            size_t keys_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(KeyType));
            d_keys_tmp2_view = temp_storage.subview(offset_bytes / sizeof(uint), keys_uint_count).template as<KeyType>();
            offset_bytes += keys_uint_count * sizeof(uint);

            offset_bytes = align_up_uint(offset_bytes, alignof(ValueType));
            size_t values_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(ValueType));
            d_values_tmp2_view = temp_storage.subview(offset_bytes / sizeof(uint), values_uint_count).template as<ValueType>();
            offset_bytes += values_uint_count * sizeof(uint);
        }

        // d_ctrs: num_portions * num_passes * uint
        size_t ctrs_count = (size_t)num_portions * num_passes;
        auto d_ctrs_view = temp_storage.subview(offset_bytes / sizeof(uint), ctrs_count);

        auto radix_sort_key = get_type_and_op_desc<KeyType, ValueType>()
                              + luisa::string(IS_DESCENDING ? "_desc" : "_asc");


        // reset keys
        using RadixSortReset        = details::RadixSortResetModule<uint>;
        using RadixSortResetKernel  = RadixSortReset::RadixSortResetKernel;
        auto ms_radix_sort_reset_it = ms_radix_sort_reset_map.find(radix_sort_key);
        if(ms_radix_sort_reset_it == ms_radix_sort_reset_map.end())
        {
            auto shader = RadixSortReset().compile(m_device);
            if (!shader) { return -1; }
            auto [it, inserted] = ms_radix_sort_reset_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_reset_it = it;
        }
        if(ms_radix_sort_reset_it == ms_radix_sort_reset_map.end()) { return -1; }
        auto ms_radix_sort_reset_ptr =
            reinterpret_cast<RadixSortResetKernel*>(&(*ms_radix_sort_reset_it->second));
        if(!ms_radix_sort_reset_ptr) { return -1; }

        cmdlist << (*ms_radix_sort_reset_ptr)(d_bins_view, 0u).dispatch(bins_count)
                << (*ms_radix_sort_reset_ptr)(d_ctrs_view, 0u).dispatch(ctrs_count);

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using RadixSortHistogramKernel  = RadixSortHistogram::RadixSortHistogramKernel;
        auto ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_key);
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            auto shader = RadixSortHistogram().compile(m_device);
            if (!shader) { return -1; }
            auto [it, inserted] = ms_radix_sort_histogram_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_histogram_it = it;
        }
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end()) { return -1; }
        auto ms_radix_sort_histogram_ptr =
            reinterpret_cast<RadixSortHistogramKernel*>(&(*ms_radix_sort_histogram_it->second));
        if(!ms_radix_sort_histogram_ptr) { return -1; }
        const auto num_sms             = BLOCK_SIZE;
        const auto histo_blocks_per_sm = 1;
        cmdlist << (*ms_radix_sort_histogram_ptr)(
                       d_bins_view, ByteBufferView{d_keys.current()}, num_items, begin_bit, end_bit)
                       .dispatch(num_sms * histo_blocks_per_sm * m_block_size);

        // exclusive scan
        using RadixSortExclusiveSum = details::RadixSortExclusiveSumModule<RADIX_DIGITS, BLOCK_SIZE, WARP_NUMS>;
        using RadixSortExclusiveSumKernel   = RadixSortExclusiveSum::RadixSortExclusiveSumKernel;
        auto ms_radix_sort_exclusive_sum_it = ms_radix_sort_exclusive_sum_map.find(radix_sort_key);
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end())
        {
            auto shader = RadixSortExclusiveSum().compile(m_device);
            if (!shader) { return -1; }
            auto [it, inserted] =
                ms_radix_sort_exclusive_sum_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_exclusive_sum_it = it;
        }
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end()) { return -1; }
        auto ms_radix_sort_exclusive_sum_ptr =
            reinterpret_cast<RadixSortExclusiveSumKernel*>(&(*ms_radix_sort_exclusive_sum_it->second));
        if(!ms_radix_sort_exclusive_sum_ptr) { return -1; }

        cmdlist << (*ms_radix_sort_exclusive_sum_ptr)(d_bins_view).dispatch(num_passes * m_block_size);

        // one sweep
        auto d_keys_tmp   = d_keys.alternate();
        auto d_values_tmp = d_values.alternate();
        if(!is_overwrite_okay && num_passes % 2 == 0)
        {
            d_keys.d_buffer[1]   = d_keys_tmp2_view;
            d_values.d_buffer[1] = d_values_tmp2_view;
        }

        using RadixSortOneSweep =
            details::RadixSortOneSweepModule<KeyType, ValueType, KEY_ONLY, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ONESWEEP_ITMES_PER_THREADS>;
        using RadixSortOneSweepKernel = RadixSortOneSweep::RadixSortOneSweepKernel;

        auto ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(radix_sort_key);
        if(ms_radix_sort_onesweep_it == ms_radix_sort_one_sweep_map.end())
        {
            auto shader = RadixSortOneSweep().compile(m_device);
            if (!shader) { return -1; }
            ms_radix_sort_one_sweep_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(radix_sort_key);
        }
        if(ms_radix_sort_onesweep_it == ms_radix_sort_one_sweep_map.end()) { return -1; }
        auto ms_radix_sort_onesweep_ptr =
            reinterpret_cast<RadixSortOneSweepKernel*>(&(*ms_radix_sort_onesweep_it->second));
        if(!ms_radix_sort_onesweep_ptr) { return -1; }

        for(uint current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            uint num_bit = std::min(end_bit - current_bit, RADIX_BITS);

            for(uint portion = 0; portion < num_portions; ++portion)
            {
                uint portion_num_items = std::min(num_items - portion * PORTION_SIZE, PORTION_SIZE);
                uint num_blocks        = ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

                cmdlist << (*ms_radix_sort_reset_ptr)(d_lookback_view, 0u)
                               .dispatch(lookback_count);

                // dispatch
                cmdlist
                    << (*ms_radix_sort_onesweep_ptr)(
                           d_lookback_view,
                           d_ctrs_view.subview(portion * num_passes + pass, 1),
                           d_bins_view.subview((portion * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS),
                           portion < num_portions - 1 ?
                               d_bins_view.subview(((portion + 1) * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS) :
                               d_bins_view.subview(0, RADIX_DIGITS),  // dummy: last portion, bins_out unused but must be valid-sized
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
                               DoubleBuffer<KeyType>(d_keys_tmp, d_keys_tmp2_view) :
                               DoubleBuffer<KeyType>(d_keys_tmp2_view, d_keys_tmp);
                d_values = num_passes % 2 == 0 ?
                               DoubleBuffer<ValueType>(d_values_tmp, d_values_tmp2_view) :
                               DoubleBuffer<ValueType>(d_values_tmp2_view, d_values_tmp);
            }
            d_keys.selector ^= 1;
            d_values.selector ^= 1;
        }

        return 0;
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_histogram_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_exclusive_sum_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_one_sweep_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_reset_map;
};
}  // namespace luisa::parallel_primitive
