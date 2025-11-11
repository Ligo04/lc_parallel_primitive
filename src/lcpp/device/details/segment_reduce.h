/*
 * @Author: Ligo 
 * @Date: 2025-11-07 14:37:01 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-11 10:52:57
 */


#pragma once
#include "luisa/dsl/resource.h"
#include "luisa/dsl/stmt.h"
#include "luisa/runtime/buffer.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/agent/agent_reduce.h>
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
    template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class SegmentReduceModule : public LuisaModule
    {
      public:
        using SegmentReduceKernel =
            Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<uint>, Buffer<uint>, uint, Type4Byte>;

        using FixedSizeSegmentReduceKernel =
            Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, uint, uint, Type4Byte>;

        template <typename ReduceOp, typename TransformOp = IdentityOp>
        using AgentReduceT =
            AgentReduce<Type4Byte, ReduceOp, TransformOp, BLOCK_SIZE, ITEMS_PER_THREAD, WARP_SIZE>;

        template <typename ReduceOp>
        U<SegmentReduceKernel> compile(Device& device, size_t shared_mem_size, ReduceOp reduce_op)
        {
            U<SegmentReduceKernel> ms_segment_reduce_shader = nullptr;

            lazy_compile(device,
                         ms_segment_reduce_shader,
                         [&](BufferVar<Type4Byte> d_arr_in,
                             BufferVar<Type4Byte> d_arr_out,
                             BufferVar<uint>      d_begin_offsets,
                             BufferVar<uint>      d_end_offsets,
                             UInt                 d_num_segments,
                             Var<Type4Byte>       initial_value) noexcept
                         {
                             set_block_size(BLOCK_SIZE);

                             UInt segment_begin = d_begin_offsets.read(block_id().x);
                             UInt segment_end   = d_end_offsets.read(block_id().x);

                             $if(segment_begin == segment_end)
                             {
                                 // empty segment
                                 $if(thread_id().x == 0)
                                 {
                                     d_arr_out.write(block_id().x, initial_value);
                                 };
                                 return;
                             };

                             SmemTypePtr<Type4Byte> smem_data = new SmemType<Type4Byte>{shared_mem_size};
                             Var<Type4Byte> block_aggregate =
                                 AgentReduceT(smem_data, d_arr_in, reduce_op).ConsumeRange(segment_begin, segment_end);

                             // normalize as needed(for keyvalue-pair)
                             //  if constexpr(is_key_value_pair_t<Type4Byte>)
                             //  {
                             //      block_aggregate = extract_value(block_aggregate);
                             //  };

                             $if(thread_id().x == 0)
                             {
                                 // finalize and store aggregate
                                 d_arr_out.write(block_id().x, reduce_op(initial_value, block_aggregate));
                             };
                         });

            return ms_segment_reduce_shader;
        }

        static constexpr auto segments_per_small_block = Policy_hub<Type4Byte>::SmallReducePolicy::SEGMENTS_PER_BLOCK;
        static constexpr auto small_threads_per_warp = Policy_hub<Type4Byte>::SmallReducePolicy::WARP_THREADS;
        static constexpr auto small_items_per_tile = Policy_hub<Type4Byte>::SmallReducePolicy::ITEMS_PER_TILE;
        static constexpr auto small_items_per_threads = Policy_hub<Type4Byte>::SmallReducePolicy::ITEMS_PER_THREAD;

        template <typename ReduceOp, typename TransformOp = IdentityOp>
        using AgentSmallReduceT =
            AgentWarpReduce<Type4Byte, ReduceOp, TransformOp, small_threads_per_warp, ITEMS_PER_THREAD>;


        template <typename ReduceOp>
        U<FixedSizeSegmentReduceKernel> compile_fixed_size(Device& device, size_t shared_mem_size, ReduceOp reduce_op)
        {
            U<FixedSizeSegmentReduceKernel> ms_fixed_size_segment_reduce_shader = nullptr;

            lazy_compile(
                device,
                ms_fixed_size_segment_reduce_shader,
                [&](BufferVar<Type4Byte> d_arr_in,
                    BufferVar<Type4Byte> d_arr_out,
                    UInt                 d_num_segments,
                    UInt                 d_segment_size,
                    Var<Type4Byte>       initial_value) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt bid  = block_id().x;
                    UInt thid = thread_id().x;

                    $if(d_segment_size <= UInt(small_items_per_tile))
                    {
                        UInt sid_within_block = thid / small_threads_per_warp;
                        UInt lane_id          = thid % small_threads_per_warp;
                        UInt global_segment_id = bid * UInt(segments_per_small_block) + sid_within_block;

                        const auto segment_begin = global_segment_id * d_segment_size;

                        $if(global_segment_id < d_num_segments)
                        {
                            $if(d_segment_size == 0)
                            {
                                $if(lane_id == 0)
                                {
                                    d_arr_out.write(global_segment_id, initial_value);
                                };
                            };

                            SmemTypePtr<Type4Byte> smem_data = new SmemType<Type4Byte>{segments_per_small_block};
                            Var<Type4Byte> warp_aggregate =
                                AgentSmallReduceT(smem_data, d_arr_in, reduce_op)
                                    .ConsumeRange(segment_begin, segment_begin + d_segment_size);
                            $if(lane_id == 0)
                            {
                                d_arr_out.write(global_segment_id, reduce_op(initial_value, warp_aggregate));
                            };
                        };
                    }
                    $else
                    {
                        // fallback to normal segment reduce
                        UInt segment_begin = bid * d_segment_size;

                        SmemTypePtr<Type4Byte> smem_data = new SmemType<Type4Byte>{shared_mem_size};
                        Var<Type4Byte>         block_aggregate =
                            AgentReduceT(smem_data, d_arr_in, reduce_op).ConsumeRange(segment_begin, segment_begin + d_segment_size);

                        $if(thid == 0)
                        {
                            // finalize and store aggregate
                            d_arr_out.write(bid, reduce_op(initial_value, block_aggregate));
                        };
                    };
                });

            return ms_fixed_size_segment_reduce_shader;
        }
    };


    template <NumericT Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE>
    class ArgSegmentReduce : public LuisaModule
    {
      public:
        using ArgConstructShaderT = Shader<1, Buffer<Type4Byte>, Buffer<IndexValuePairT<Type4Byte>>>;

        using ArgAssignShaderT =
            Shader<1, Buffer<IndexValuePairT<Type4Byte>>, Buffer<Type4Byte>, Buffer<uint>>;

        U<ArgConstructShaderT> compile_arg_construct_shader(Device& device)
        {
            U<ArgConstructShaderT> ms_arg_construct_shader = nullptr;
            lazy_compile(device,
                         ms_arg_construct_shader,
                         [&](BufferVar<Type4Byte>                  arr_in,
                             BufferVar<uint>                       d_begin_offsets,
                             BufferVar<uint>                       d_end_offset,
                             BufferVar<IndexValuePairT<Type4Byte>> g_kv_out)
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt global_id = block_id().x * block_size().x + thread_id().x;

                             Var<IndexValuePairT<Type4Byte>> initial{0, 0};
                             initial.key   = global_id - d_begin_offsets.read(global_id);
                             initial.value = arr_in.read(global_id);
                             g_kv_out.write(global_id, initial);
                         });
            return ms_arg_construct_shader;
        }

        U<ArgAssignShaderT> compile_arg_assign_shader(Device& device)
        {
            U<ArgAssignShaderT> ms_arg_assign_shader = nullptr;
            lazy_compile(device,
                         ms_arg_assign_shader,
                         [&](BufferVar<IndexValuePairT<Type4Byte>> kvp_in,
                             BufferVar<Type4Byte>                  value_out,
                             BufferVar<uint>                       index_out)
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt global_id = block_id().x * block_size().x + thread_id().x;

                             Var<IndexValuePairT<Type4Byte>> kvp = kvp_in.read(global_id);
                             index_out.write(global_id, kvp.key);
                             value_out.write(global_id, kvp.value);
                         });
            return ms_arg_assign_shader;
        }
    };

}  // namespace details
}  // namespace luisa::parallel_primitive