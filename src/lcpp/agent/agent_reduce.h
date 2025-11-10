/*
 * @Author: Ligo
 * @Date: 2025-11-07 16:45:23
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 18:03:08
 */

#pragma once
#include "lcpp/block/block_reduce.h"
#include "lcpp/thread/thread_reduce.h"
#include "lcpp/warp/warp_reduce.h"
#include "luisa/dsl/resource.h"
#include <cstddef>
#include <lcpp/common/grid_even_shared.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <NumericT Type4Byte, typename ReduceOp, typename TransformOp, typename CollectiveReduceT, bool IsWarpReduction, size_t NUM_THREADS, size_t ITEMS_PER_THREAD>
    class AgentReduceImpl : public LuisaModule
    {
      public:
        AgentReduceImpl(SmemTypePtr<Type4Byte>& smem_data,
                        BufferVar<Type4Byte>&   d_in,
                        ReduceOp                reduce_op,
                        TransformOp             transform_op,
                        UInt                    land_id)
            : m_smem_data(smem_data)
            , m_wrapped_in(d_in)
            , m_reduce_op(reduce_op)
            , m_transform_op(transform_op)
            , m_land_id(land_id) {};

      public:
        Var<Type4Byte> ConsumeRange(UInt block_offset, UInt block_end)
        {
            Var<GridEvenShared> even_shared;
            even_shared->BlockInit<NUM_THREADS * ITEMS_PER_THREAD>(block_offset, block_end);
            return ConsumeRange(even_shared);
        }

        Var<Type4Byte> ConsumeRange(Var<GridEvenShared>& even_shared)
        {
            Var<Type4Byte> thread_aggregate;
            UInt           TILE_ITEMS = UInt(NUM_THREADS * ITEMS_PER_THREAD);
            Var<Type4Byte> result;
            $if(even_shared.block_end - even_shared.block_offset < TILE_ITEMS)
            {
                // empty segment
                thread_aggregate = def<Type4Byte>(0);
                UInt valid_items = even_shared.block_end - even_shared.block_offset;
                ConsumePartialTile<true>(thread_aggregate, even_shared.block_offset, valid_items);

                if constexpr(IsWarpReduction)
                {
                    valid_items = select(valid_items, UInt(NUM_THREADS), UInt(NUM_THREADS) <= valid_items);
                }
                result = CollectiveReduceT().Reduce(thread_aggregate, m_reduce_op, valid_items);
            }
            $else
            {
                ConsumeFullTileRange(thread_aggregate, even_shared);

                result = CollectiveReduceT().Reduce(thread_aggregate, m_reduce_op);
            };
            return result;
        }

        template <bool IsFirstTile>
        void ConsumePartialTile(Var<Type4Byte>& thread_aggregate, UInt block_offset, UInt valid_items)
        {
            UInt thread_offset = m_land_id;

            if constexpr(IsFirstTile)
            {
                $if(thread_offset < valid_items)
                {
                    thread_aggregate = m_transform_op(m_wrapped_in.read(thread_offset + block_offset));
                    thread_offset += UInt(NUM_THREADS);
                };
            }

            $while(thread_offset < valid_items)
            {
                // load data
                Var<Type4Byte> data = m_wrapped_in.read(thread_offset + block_offset);
                thread_aggregate    = m_reduce_op(thread_aggregate, m_transform_op(data));
                thread_offset += UInt(NUM_THREADS);
            };
        }

        template <bool IsFirstTile>
        void ConsumeFullTile(Var<Type4Byte>& thread_aggregate, UInt block_offset)
        {
            ArrayVar<Type4Byte, ITEMS_PER_THREAD> items;
            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                items[i] = m_transform_op(m_wrapped_in.read(block_offset + m_land_id + i * UInt(NUM_THREADS)));
            };

            if constexpr(IsFirstTile)
            {
                thread_aggregate = ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(items, m_reduce_op);
            }
            else
            {
                thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(items, m_reduce_op, thread_aggregate);
            }
        }

      private:
        void ConsumeFullTileRange(Var<Type4Byte>& thread_aggregate, Var<GridEvenShared>& even_shared)
        {
            ConsumeFullTile<true>(thread_aggregate, even_shared.block_offset);

            $if(even_shared.block_end - even_shared.block_offset < even_shared.block_stride)
            {
                return;
            };
            even_shared.block_offset += even_shared.block_stride;

            $while(even_shared.block_offset <= even_shared.block_end - UInt(NUM_THREADS * ITEMS_PER_THREAD))
            {
                ConsumeFullTile<false>(thread_aggregate, even_shared.block_offset);
                $if(even_shared.block_end - even_shared.block_offset < even_shared.block_stride)
                {
                    return;
                };
                even_shared.block_offset += even_shared.block_stride;
            };

            $if(even_shared.block_offset < even_shared.block_end)
            {
                UInt valid_items = even_shared.block_end - even_shared.block_offset;
                ConsumePartialTile<false>(thread_aggregate, even_shared.block_offset, valid_items);
            };
        }

      private:
        SmemTypePtr<Type4Byte> m_smem_data;
        TransformOp            m_transform_op;
        ReduceOp               m_reduce_op;
        UInt                   m_land_id;

        BufferVar<Type4Byte>& m_wrapped_in;
    };

    template <typename Type4Byte, typename ReduceOp, typename TransformOp, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class AgentReduce
        : public AgentReduceImpl<Type4Byte, ReduceOp, TransformOp, BlockReduce<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, WARP_SIZE>, false, BLOCK_SIZE, ITEMS_PER_THREAD>
    {
      public:
        using Base =
            AgentReduceImpl<Type4Byte, ReduceOp, TransformOp, BlockReduce<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, WARP_SIZE>, false, BLOCK_SIZE, ITEMS_PER_THREAD>;
        AgentReduce(SmemTypePtr<Type4Byte>& smem_data,
                    BufferVar<Type4Byte>&   in,
                    ReduceOp                reduce_op,
                    TransformOp             transform_op = IdentityOp())
            : Base(smem_data, in, reduce_op, transform_op, thread_id().x) {};
    };

    template <typename Type4Byte, typename ReduceOp, typename TransformOp, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class AgentWarpReduce
        : public AgentReduceImpl<Type4Byte, ReduceOp, TransformOp, WarpReduce<Type4Byte, WARP_NUMS>, true, WARP_NUMS, ITEMS_PER_THREAD>
    {
      public:
        using Base =
            AgentReduceImpl<Type4Byte, ReduceOp, TransformOp, WarpReduce<Type4Byte, WARP_NUMS>, true, WARP_NUMS, ITEMS_PER_THREAD>;
        AgentWarpReduce(SmemTypePtr<Type4Byte>& smem_data,
                        BufferVar<Type4Byte>&   in,
                        ReduceOp                reduce_op,
                        TransformOp             transform_op = IdentityOp())
            : Base(smem_data, in, reduce_op, transform_op, thread_id().x % UInt(WARP_NUMS)) {};
    };
}  // namespace details


}  // namespace luisa::parallel_primitive