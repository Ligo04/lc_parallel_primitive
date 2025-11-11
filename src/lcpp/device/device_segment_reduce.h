/*
 * @Author: Ligo 
 * @Date: 2025-11-07 14:17:58 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 21:27:54
 */
#pragma once

#include "details/segment_reduce.h"
#include "lcpp/common/utils.h"
#include "lcpp/device/details/reduce.h"
#include "luisa/core/basic_traits.h"
#include "luisa/core/logging.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/runtime/core.h>


namespace luisa::parallel_primitive
{
using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceSegmentReduce : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

  public:
    DeviceSegmentReduce()  = default;
    ~DeviceSegmentReduce() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                uint                  num_segments,
                BufferView<uint>      d_begin_offsets,
                BufferView<uint>      d_end_offsets,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        segment_reduce_array_recursive<Type4Byte>(
            cmdlist, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, reduce_op, initial_value);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                uint                  num_segments,
                uint                  segment_size,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        fixed_segment_reduce_array_recursive<Type4Byte>(
            cmdlist, d_in, d_out, num_segments, segment_size, reduce_op, initial_value);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            0);
    }

    template <NumericT Type4Byte>
    void Sum(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_segments,
            segment_size,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            0);
    }


    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(cmdlist,
               stream,
               d_in,
               d_out,
               num_segments,
               d_begin_offsets,
               d_end_offsets,
               compute::min,
               std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Min(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(cmdlist, stream, d_in, d_out, num_segments, segment_size, compute::min, std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             Stream&               stream,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(cmdlist,
               stream,
               d_in,
               d_out,
               num_segments,
               d_begin_offsets,
               d_end_offsets,
               compute::max,
               std::numeric_limits<Type4Byte>::lowest());
    }

    template <NumericT Type4Byte>
    void Max(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(cmdlist, stream, d_in, d_out, num_segments, segment_size, compute::max, std::numeric_limits<Type4Byte>::lowest());
    }


    template <NumericT ValueType>
    void ArgMax(CommandList&          cmdlist,
                Stream&               stream,
                BufferView<ValueType> d_in,
                BufferView<ValueType> d_out,
                BufferView<uint>      d_index_out,
                uint                  num_segments,
                BufferView<uint>      d_begin_offsets,
                BufferView<uint>      d_end_offsets)
    {
    }


  private:
    template <NumericT Type4Byte, typename ReduceOp>
    void segment_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                        BufferView<Type4Byte>        arr_in,
                                        BufferView<Type4Byte>        arr_out,
                                        size_t                       num_segments,
                                        BufferView<uint>             d_begin_offsets,
                                        BufferView<uint>             d_end_offsets,
                                        ReduceOp                     reduce_op,
                                        Type4Byte                    initial_value)
    {
        using SegmentReduce = details::SegmentReduceModule<Type4Byte, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using SegmentReduceKernel = SegmentReduce::SegmentReduceKernel;

        auto key                  = get_type_and_op_desc<Type4Byte>(reduce_op, IdentityOp());
        auto ms_segment_reduce_it = ms_segment_reduce_map.find(key);
        if(ms_segment_reduce_it == ms_segment_reduce_map.end())
        {
            auto shader = SegmentReduce().compile(m_device, m_shared_mem_size, reduce_op);
            ms_segment_reduce_map.try_emplace(key, std::move(shader));
            ms_segment_reduce_it = ms_segment_reduce_map.find(key);
        }
        auto ms_segment_reduce_ptr = reinterpret_cast<SegmentReduceKernel*>(&(*ms_segment_reduce_it->second));
        cmdlist << (*ms_segment_reduce_ptr)(arr_in, arr_out, d_begin_offsets, d_end_offsets, num_segments, initial_value)
                       .dispatch(num_segments * m_block_size);
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void fixed_segment_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                              BufferView<Type4Byte>        arr_in,
                                              BufferView<Type4Byte>        arr_out,
                                              uint                         num_segments,
                                              uint                         segment_size,
                                              ReduceOp                     reduce_op,
                                              Type4Byte                    initial_value)
    {

        using SegmentReduce = details::SegmentReduceModule<Type4Byte, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using FixedSizeSegmentReduceKernel = SegmentReduce::FixedSizeSegmentReduceKernel;

        uint segment_per_block = 1;
        if(segment_size <= SegmentReduce::small_items_per_tile)
        {
            segment_per_block = SegmentReduce::segments_per_small_block;
        }

        const auto num_segments_per_invocation = static_cast<uint>(std::numeric_limits<int32_t>::max());
        const auto num_invocations = ceil_div(num_segments, num_segments_per_invocation);
        LUISA_INFO("Fixed Segment Reduce: num_segments={}, segment_size={}, segment_per_block={}, num_invocations={}",
                   num_segments,
                   segment_size,
                   segment_per_block,
                   num_invocations);

        auto key = get_type_and_op_desc<Type4Byte>(reduce_op, IdentityOp());
        auto ms_fixed_size_segment_reduce_it = ms_fixed_segment_reduce_map.find(key);
        if(ms_fixed_size_segment_reduce_it == ms_fixed_segment_reduce_map.end())
        {
            auto shader = SegmentReduce().compile_fixed_size(m_device, m_shared_mem_size, reduce_op);
            ms_fixed_segment_reduce_map.try_emplace(key, std::move(shader));
            ms_fixed_size_segment_reduce_it = ms_fixed_segment_reduce_map.find(key);
        }
        auto ms_fixed_size_segment_reduce_ptr =
            reinterpret_cast<FixedSizeSegmentReduceKernel*>(&(*ms_fixed_size_segment_reduce_it->second));
        for(auto invocation_index = 0u; invocation_index < num_invocations; invocation_index++)
        {
            const auto current_seg_offset = invocation_index * num_segments_per_invocation;
            const auto num_current_segments =
                std::min(num_segments - current_seg_offset, num_segments_per_invocation);

            const auto num_current_blocks = ceil_div(num_current_segments, segment_per_block);
            // LUISA_INFO("Dispatching fixed segment reduce: invocation_index={}, current_seg_offset={}, num_current_segments={}, num_current_blocks={}",
            //            invocation_index,
            //            current_seg_offset,
            //            num_current_segments,
            //            num_current_blocks);
            cmdlist << (*ms_fixed_size_segment_reduce_ptr)(arr_in, arr_out, num_current_segments, segment_size, initial_value)
                           .dispatch(num_current_blocks * m_block_size);
        }
    }


    template <NumericT Type4Byte>
    void arg_construct(CommandList& cmdlist, BufferView<Type4Byte> d_in, BufferView<IndexValuePairT<Type4Byte>> d_kv_out)
    {
        using ArgReduce          = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgConstructShader = ArgReduce::ArgConstructShaderT;
        auto key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            auto shader = ArgReduce().compile_arg_construct_shader(m_device);
            ms_arg_construct_map.try_emplace(key, std::move(shader));
            ms_arg_construct_it = ms_arg_construct_map.find(key);
        }
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShader*>(&(*ms_arg_construct_it->second));
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(d_in.size());
    }

    template <NumericT Type4Byte>
    void arg_assign(CommandList&                           cmdlist,
                    BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                    BufferView<Type4Byte>                  d_value_out,
                    BufferView<uint>                       d_index_out)
    {
        using ArgReduce       = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgAssignShader = ArgReduce::ArgAssignShaderT;
        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_assign_shader(m_device);
            ms_arg_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_assign_map.find(key);
        }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_value_out, d_index_out).dispatch(d_index_out.size());
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_segment_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_fixed_segment_reduce_map;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
};
}  // namespace luisa::parallel_primitive