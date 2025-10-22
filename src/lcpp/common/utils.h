/*
 * @Author: Ligo 
 * @Date: 2025-10-22 17:17:43 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 18:24:53
 */


#pragma once
#include <cmath>
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
static inline bool is_power_of_two(int x)
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
    luisa::compute::Var<Type4Byte> result;
    luisa::compute::Int            src_lane = curr_lane_id - offset;
    $if(src_lane >= first_lane)
    {
        result = compute::warp_read_lane(input, src_lane);
        // compute::device_log("thid:{}, src_lane: {}, input: {}, result: {}",
        //                     compute::dispatch_id().x,
        //                     src_lane,
        //                     input,
        //                     result);
    }
    $else
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


template <size_t log_mem_banks = 5>
inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
{
    return i >> log_mem_banks;
}
};  // namespace luisa::parallel_primitive