/*
 * @Author: Ligo 
 * @Date: 2025-11-07 14:17:58 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 21:27:54
 */
#pragma once

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
#include <lcpp/device/details/segment_reduce.h>
#include <lcpp/common/utils.h>

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
        m_created                  = true;
    }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for Reduce / Sum / Min / Max
    /// Segment reduce dispatches a single kernel, no temp storage needed.
    /// Returns 0; API kept for consistency with other Device modules.
    template <typename Type4Byte>
    static size_t GetTempStorageBytes(size_t /*num_item*/)
    {
        return 0;
    }

    /// Temp storage bytes for ArgMin / ArgMax
    /// Needs: d_in_kv(num_item) + d_out_kv(num_segments)
    template <typename Type4Byte>
    static size_t GetArgTempStorageBytes(size_t num_item, size_t num_segments)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t bytes = 0;
        bytes += num_item * sizeof(IVP);            // d_in_kv
        bytes  = align_up_uint(bytes, alignof(IVP));
        bytes += num_segments * sizeof(IVP);        // d_out_kv
        return bytes;
    }

    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <typename Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                BufferView<uint>      /*temp_storage*/,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                uint                  num_segments,
                BufferView<uint>      d_begin_offsets,
                BufferView<uint>      d_end_offsets,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        lcpp_check(segment_reduce_array_recursive<Type4Byte>(
            cmdlist, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, reduce_op, initial_value), cmdlist, debug_stream());
    }

    template <typename Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                BufferView<uint>      /*temp_storage*/,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                uint                  num_segments,
                uint                  segment_size,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        lcpp_check(
            fixed_segment_reduce_array_recursive<Type4Byte>(
            cmdlist, d_in, d_out, num_segments, segment_size, reduce_op, initial_value)   ,
        cmdlist, debug_stream());
    }

    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, SumOp(), Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void Sum(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_segments, segment_size, SumOp(), Type4Byte(0));
    }


    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(cmdlist,
               temp_storage,
               d_in,
               d_out,
               num_segments,
               d_begin_offsets,
               d_end_offsets,
               compute::min,
               std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Min(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_segments, segment_size, compute::min, std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             uint                  num_segments,
             BufferView<uint>      d_begin_offsets,
             BufferView<uint>      d_end_offsets)
    {
        Reduce(cmdlist,
               temp_storage,
               d_in,
               d_out,
               num_segments,
               d_begin_offsets,
               d_end_offsets,
               compute::max,
               std::numeric_limits<Type4Byte>::lowest());
    }

    template <NumericT Type4Byte>
    void Max(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, uint num_segments, uint segment_size)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_segments, segment_size, compute::max, std::numeric_limits<Type4Byte>::lowest());
    }


    template <NumericT ValueType>
    void ArgMax(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<ValueType> d_in,
                BufferView<ValueType> d_out,
                BufferView<uint>      d_index_out,
                uint                  num_segments,
                BufferView<uint>      d_begin_offsets,
                BufferView<uint>      d_end_offsets)
    {
        using IVP = IndexValuePairT<ValueType>;

        // slice kv buffers from temp_storage
        size_t offset_uint = 0;
        size_t in_kv_uint_count  = bytes_to_uint_count(d_in.size() * sizeof(IVP));
        auto   d_in_kv           = temp_storage.subview(offset_uint, in_kv_uint_count).template as<IVP>();
        offset_uint += in_kv_uint_count;
        size_t out_kv_uint_count = bytes_to_uint_count(num_segments * sizeof(IVP));
        auto   d_out_kv          = temp_storage.subview(offset_uint, out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct(cmdlist, d_in, d_in_kv, num_segments), cmdlist, debug_stream());
        Reduce<IVP>(
            cmdlist,
            BufferView<uint>{},
            d_in_kv,
            d_out_kv,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            ArgMaxOp(),
            IVP{1, std::numeric_limits<ValueType>::min()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<ValueType>(cmdlist, d_out_kv, d_begin_offsets, d_index_out, d_out, num_segments),
        cmdlist, debug_stream());
    }


    template <NumericT ValueType>
    void ArgMax(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<ValueType> d_in,
                BufferView<ValueType> d_out,
                BufferView<uint>      d_index_out,
                uint                  num_segments,
                uint                  segment_size)
    {
        using IVP = IndexValuePairT<ValueType>;

        // slice kv buffers from temp_storage
        size_t offset_uint = 0;
        size_t in_kv_uint_count  = bytes_to_uint_count(d_in.size() * sizeof(IVP));
        auto   d_in_kv           = temp_storage.subview(offset_uint, in_kv_uint_count).template as<IVP>();
        offset_uint += in_kv_uint_count;
        size_t out_kv_uint_count = bytes_to_uint_count(num_segments * sizeof(IVP));
        auto   d_out_kv          = temp_storage.subview(offset_uint, out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct(cmdlist, d_in, d_in_kv, num_segments), cmdlist, debug_stream());

        Reduce<IVP>(
            cmdlist,
            BufferView<uint>{},
            d_in_kv,
            d_out_kv,
            num_segments,
            segment_size,
            ArgMaxOp(),
            IVP{1, std::numeric_limits<ValueType>::min()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_fixed_size_assign<ValueType>(cmdlist, d_out_kv, d_index_out, d_out, num_segments, segment_size),
        cmdlist, debug_stream());
    }


    template <NumericT ValueType>
    void Argmin(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<ValueType> d_in,
                BufferView<ValueType> d_out,
                BufferView<uint>      d_index_out,
                uint                  num_segments,
                BufferView<uint>      d_begin_offsets,
                BufferView<uint>      d_end_offsets)
    {
        using IVP = IndexValuePairT<ValueType>;

        // slice kv buffers from temp_storage
        size_t offset_uint = 0;
        size_t in_kv_uint_count  = bytes_to_uint_count(d_in.size() * sizeof(IVP));
        auto   d_in_kv           = temp_storage.subview(offset_uint, in_kv_uint_count).template as<IVP>();
        offset_uint += in_kv_uint_count;
        size_t out_kv_uint_count = bytes_to_uint_count(num_segments * sizeof(IVP));
        auto   d_out_kv          = temp_storage.subview(offset_uint, out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct(cmdlist, d_in, d_in_kv, num_segments), cmdlist, debug_stream());

        Reduce<IVP>(
            cmdlist,
            BufferView<uint>{},
            d_in_kv,
            d_out_kv,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            ArgMinOp(),
            IVP{1, std::numeric_limits<ValueType>::max()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<ValueType>(cmdlist, d_out_kv, d_begin_offsets, d_index_out, d_out, num_segments),
        cmdlist, debug_stream());
    }


    template <NumericT ValueType>
    void ArgMin(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<ValueType> d_in,
                BufferView<ValueType> d_out,
                BufferView<uint>      d_index_out,
                uint                  num_segments,
                uint                  segment_size)
    {
        using IVP = IndexValuePairT<ValueType>;

        // slice kv buffers from temp_storage
        size_t offset_uint = 0;
        size_t in_kv_uint_count  = bytes_to_uint_count(d_in.size() * sizeof(IVP));
        auto   d_in_kv           = temp_storage.subview(offset_uint, in_kv_uint_count).template as<IVP>();
        offset_uint += in_kv_uint_count;
        size_t out_kv_uint_count = bytes_to_uint_count(num_segments * sizeof(IVP));
        auto   d_out_kv          = temp_storage.subview(offset_uint, out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct(cmdlist, d_in, d_in_kv, num_segments), cmdlist, debug_stream());

        Reduce<IVP>(
            cmdlist,
                BufferView<uint>{},
                d_in_kv,
                d_out_kv,
                num_segments,
                segment_size,
                ArgMinOp(),
                IVP{1, std::numeric_limits<ValueType>::max()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_fixed_size_assign<ValueType>(cmdlist, d_out_kv, d_index_out, d_out, num_segments, segment_size),
        cmdlist, debug_stream());
    }


  private:
    template <typename Type, typename ReduceOp>
    [[nodiscard]] int segment_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                        BufferView<Type>             arr_in,
                                        BufferView<Type>             arr_out,
                                        size_t                       num_segments,
                                        BufferView<uint>             d_begin_offsets,
                                        BufferView<uint>             d_end_offsets,
                                        ReduceOp                     reduce_op,
                                        Type                         initial_value)
    {
        using SegmentReduce = details::SegmentReduceModule<Type, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using SegmentReduceKernel = SegmentReduce::SegmentReduceKernel;

        auto key                  = get_type_and_op_desc<Type>(reduce_op);
        auto ms_segment_reduce_it = ms_segment_reduce_map.find(key);
        if(ms_segment_reduce_it == ms_segment_reduce_map.end())
        {
            auto shader = SegmentReduce().compile(m_device, m_shared_mem_size, reduce_op);
            if (!shader) { return -1; }
            ms_segment_reduce_map.try_emplace(key, std::move(shader));
            ms_segment_reduce_it = ms_segment_reduce_map.find(key);
        }
        if(ms_segment_reduce_it == ms_segment_reduce_map.end()) { return -1; }
        auto ms_segment_reduce_ptr = reinterpret_cast<SegmentReduceKernel*>(&(*ms_segment_reduce_it->second));
        if(!ms_segment_reduce_ptr) { return -1; }
        cmdlist << (*ms_segment_reduce_ptr)(arr_in, arr_out, d_begin_offsets, d_end_offsets, num_segments, initial_value)
                       .dispatch(num_segments * m_block_size);
        return 0;
    }

    template <typename Type4Byte, typename ReduceOp>
    [[nodiscard]] int fixed_segment_reduce_array_recursive(luisa::compute::CommandList& cmdlist,
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

        auto key                             = get_type_and_op_desc<Type4Byte>(reduce_op);
        auto ms_fixed_size_segment_reduce_it = ms_fixed_segment_reduce_map.find(key);
        if(ms_fixed_size_segment_reduce_it == ms_fixed_segment_reduce_map.end())
        {
            auto shader = SegmentReduce().compile_fixed_size(m_device, m_shared_mem_size, reduce_op);
            if (!shader) { return -1; }
            ms_fixed_segment_reduce_map.try_emplace(key, std::move(shader));
            ms_fixed_size_segment_reduce_it = ms_fixed_segment_reduce_map.find(key);
        }
        if(ms_fixed_size_segment_reduce_it == ms_fixed_segment_reduce_map.end()) { return -1; }
        auto ms_fixed_size_segment_reduce_ptr =
            reinterpret_cast<FixedSizeSegmentReduceKernel*>(&(*ms_fixed_size_segment_reduce_it->second));
        if(!ms_fixed_size_segment_reduce_ptr) { return -1; }
        for(auto invocation_index = 0u; invocation_index < num_invocations; invocation_index++)
        {
            const auto current_seg_offset = invocation_index * num_segments_per_invocation;
            const auto num_current_segments =
                std::min(num_segments - current_seg_offset, num_segments_per_invocation);

            const auto num_current_blocks = ceil_div(num_current_segments, segment_per_block);
            cmdlist << (*ms_fixed_size_segment_reduce_ptr)(arr_in, arr_out, num_current_segments, segment_size, initial_value)
                           .dispatch(num_current_blocks * m_block_size);
        }
        return 0;
    }


    template <NumericT Type4Byte>
    [[nodiscard]] int arg_construct(CommandList&                           cmdlist,
                       BufferView<Type4Byte>                  d_in,
                       BufferView<IndexValuePairT<Type4Byte>> d_kv_out,
                       uint                                   num_segments)
    {
        using ArgReduce          = details::ArgSegmentReduceModule<Type4Byte, BLOCK_SIZE>;
        using ArgConstructShader = ArgReduce::ArgConstructShaderT;
        auto key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            auto shader = ArgReduce().compile_arg_construct_shader(m_device);
            if (!shader) { return -1; }
            ms_arg_construct_map.try_emplace(key, std::move(shader));
            ms_arg_construct_it = ms_arg_construct_map.find(key);
        }
        if(ms_arg_construct_it == ms_arg_construct_map.end()) { return -1; }
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShader*>(&(*ms_arg_construct_it->second));
        if(!ms_arg_construct_ptr) { return -1; }

        auto num_tiles = ceil_div(num_segments, m_block_size);
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(d_in.size());
        return 0;
    }

    template <NumericT Type4Byte>
    [[nodiscard]] int arg_assign(CommandList&                           cmdlist,
                    BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                    BufferView<uint>                       d_begin_offset,
                    BufferView<uint>                       d_index_out,
                    BufferView<Type4Byte>                  d_value_out,
                    uint                                   num_segments)
    {
        using ArgReduce       = details::ArgSegmentReduceModule<Type4Byte, WARP_NUMS>;
        using ArgAssignShader = ArgReduce::ArgAssignShaderT;
        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_assign_shader(m_device);
            if (!shader) { return -1; }
            ms_arg_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_assign_map.find(key);
        }
        if(ms_arg_assign_it == ms_arg_assign_map.end()) { return -1; }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        if(!ms_arg_assign_ptr) { return -1; }
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_begin_offset, d_index_out, d_value_out).dispatch(num_segments * WARP_NUMS);
        return 0;
    }

    template <NumericT Type4Byte>
    [[nodiscard]] int arg_fixed_size_assign(CommandList&                           cmdlist,
                               BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                               BufferView<uint>                       d_index_out,
                               BufferView<Type4Byte>                  d_value_out,
                               uint                                   num_segments,
                               uint                                   segment_size)
    {
        using ArgReduce       = details::ArgSegmentReduceModule<Type4Byte, WARP_NUMS>;
        using ArgAssignShader = ArgReduce::ArgFixedSizeAssignShaderT;

        const auto num_segments_per_invocation = static_cast<uint>(std::numeric_limits<int32_t>::max());
        const auto num_invocations = ceil_div(num_segments, num_segments_per_invocation);

        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_fixed_size_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_fixed_size_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_fixed_size_assign_shader(m_device);
            if (!shader) { return -1; }
            ms_arg_fixed_size_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_fixed_size_assign_map.find(key);
        }
        if(ms_arg_assign_it == ms_arg_fixed_size_assign_map.end()) { return -1; }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        if(!ms_arg_assign_ptr) { return -1; }
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, segment_size, d_index_out, d_value_out).dispatch(num_segments * WARP_NUMS);
        return 0;
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_segment_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_fixed_segment_reduce_map;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_fixed_size_assign_map;
};
}  // namespace luisa::parallel_primitive
