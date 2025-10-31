/*
 * @Author: Ligo 
 * @Date: 2025-10-22 17:17:43 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 18:24:53
 */


#pragma once
#include "lcpp/runtime/core.h"
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/stmt.h"
#include <__msvc_iter_core.hpp>
#include <cmath>
#include <cstddef>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/keyvaluepair.h>

namespace luisa::parallel_primitive
{

static inline float to_radius(float degree)
{
    return degree * 0.0174532925f;
}
static inline int imax(int a, int b)
{
    return a > b ? a : b;
}
static constexpr inline bool is_power_of_two(int x)
{
    return (x & (x - 1)) == 0;
}
static inline float radians(float degree)
{
    return degree * 0.017453292519943295769236907684886f;
}
static inline int floor_pow_2(int n)
{
#ifdef WIN32
    return 1 << (int)logb((float)n);
#else
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

static void get_temp_size_scan(size_t& temp_storage_size,
                               size_t  m_block_size,
                               size_t  items_per_thread,
                               size_t  num_items)
{
    auto         block_size       = m_block_size;
    unsigned int max_num_elements = num_items;
    temp_storage_size             = 0;
    unsigned int num_elements     = max_num_elements;  // input segment size
    int          level            = 0;
    do
    {
        // output segment size
        unsigned int num_blocks =
            imax(1, (int)ceil((float)num_elements / (items_per_thread * block_size)));
        if(num_blocks > 1)
        {
            level++;
            temp_storage_size += num_blocks;
        }
        num_elements = num_blocks;
    } while(num_elements > 1);
    temp_storage_size += 1;
}


static inline auto bit_log2(luisa::compute::UInt x)
{
    return 31 - luisa::compute::clz(x);
}

template <NumericT Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleUp(luisa::compute::Var<Type4Byte>& input,
                                         luisa::compute::UInt curr_lane_id,
                                         luisa::compute::UInt offset,
                                         luisa::compute::UInt first_lane = 0u)
{
    luisa::compute::Var<Type4Byte> result =
        compute::warp_read_lane(input, curr_lane_id - offset);

    $if(compute::Int(curr_lane_id - offset) < compute::Int(first_lane))
    {
        result = input;
    };
    return result;
};


template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> ShuffleUp(
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input,
    luisa::compute::UInt                                   curr_lane_id,
    luisa::compute::UInt                                   offset,
    luisa::compute::UInt                                   first_lane = 0u)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    luisa::compute::UInt src_lane = curr_lane_id - offset;
    $if(src_lane >= first_lane)
    {
        result.key   = compute::warp_read_lane(input.key, src_lane);
        result.value = compute::warp_read_lane(input.value, src_lane);
    }
    $else
    {
        result = input;
    };
    return result;
};

template <NumericT Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleDown(luisa::compute::Var<Type4Byte>& input,
                                           luisa::compute::UInt curr_lane_id,
                                           luisa::compute::UInt offset,
                                           luisa::compute::UInt last_lane = 32u)
{
    luisa::compute::UInt src_lane = curr_lane_id + offset;
    luisa::compute::Var<Type4Byte> result = compute::warp_read_lane(input, src_lane);
    $if(src_lane > last_lane)
    {
        result = input;
    };
    return result;
};


template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> ShuffleDown(
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input,
    luisa::compute::UInt                                   curr_lane_id,
    luisa::compute::UInt                                   offset,
    luisa::compute::UInt                                   last_lane = 32u)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    luisa::compute::UInt src_lane = curr_lane_id + offset;
    result.key   = compute::warp_read_lane(input.key, src_lane);
    result.value = compute::warp_read_lane(input.value, src_lane);
    $if(src_lane > last_lane)
    {
        result = input;
    };
    return result;
};

template <size_t log_mem_banks = 5>
inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
{
    return i >> log_mem_banks;
}

inline luisa::compute::UInt get_lane_mask_ge(luisa::compute::UInt lane_id,
                                             luisa::compute::UInt wave_size)
{
    luisa::compute::ULong mask64 = ~((1ull << lane_id) - 1ull);
    mask64 &= (1ull << wave_size) - 1ull;
    return static_cast<luisa::compute::UInt>(mask64);
}

inline luisa::compute::UInt get_lane_mask_le(luisa::compute::UInt lane_id)
{
    return (1u << (lane_id + 1)) - 1u;
}

template <size_t LOGIC_WARP_SIZE>
inline luisa::compute::UInt warp_mask(luisa::compute::UInt warp_id)
{
    constexpr bool is_pow_of_two = is_power_of_two(LOGIC_WARP_SIZE);
    constexpr bool is_arch_warp  = (LOGIC_WARP_SIZE == details::WARP_SIZE);

    luisa::compute::UInt member_mask = 0xFFFFFFFFu >> (details::WARP_SIZE - LOGIC_WARP_SIZE);

    if constexpr(is_pow_of_two && !is_arch_warp)
    {
        member_mask <<= warp_id * luisa::compute::UInt(LOGIC_WARP_SIZE);
    };

    return member_mask;
}

};  // namespace luisa::parallel_primitive