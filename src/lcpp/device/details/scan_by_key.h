/*
 * @Author: Ligo 
 * @Date: 2025-11-06 14:59:56 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:27:10
 */

#pragma once
#include "luisa/core/mathematics.h"
#include "luisa/dsl/stmt.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/runtime/buffer.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/device/details/single_pass_scan_operator.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT KeyType, NumericT ValueType, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class ScanByKeyModule : public LuisaModule
    {
      public:
        using FlagValuePairT = KeyValuePair<int, ValueType>;

        using ScanTileState = ScanTileState<FlagValuePairT>;

        using ScanByKeyKernel =
            Shader<1, Buffer<ScanTileState>, Buffer<KeyType>, Buffer<KeyType>, Buffer<ValueType>, Buffer<ValueType>, ValueType, uint>;

        using ScanTileStateInitKernel =
            Shader<1, Buffer<ScanTileState>, Buffer<KeyType>, Buffer<KeyType>, int>;


        U<ScanTileStateInitKernel> compile_scan_tile_state_init(Device& device)
        {
            U<ScanTileStateInitKernel> ms_scan_tile_state_init_shader = nullptr;
            lazy_compile(device,
                         ms_scan_tile_state_init_shader,
                         [](BufferVar<ScanTileState> tile_state,
                            BufferVar<KeyType>       d_keys_in,
                            BufferVar<KeyType>       d_prev_keys_output,
                            Int                      num_tiles) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             ScanTileStateViewer::InitializeWardStatus(tile_state, num_tiles);
                             UInt tid        = dispatch_id().x;
                             UInt tile_items = UInt(ITEMS_PER_THREAD) * block_size_x();
                             UInt tile_start = tid * tile_items;

                             $if(tid > 0 & tid < num_tiles)
                             {
                                 d_prev_keys_output.write(tid, d_keys_in.read(tile_start - 1));
                             };
                         });

            return ms_scan_tile_state_init_shader;
        }

        template <bool is_inclusive, typename ScanOp>
        U<ScanByKeyKernel> compile(Device& device, size_t shared_mem_size, ScanOp scan_op)
        {
            ScanBySegmentOp<ScanOp> pair_scan_op{scan_op};

            using TilePrefixOpT = TilePrefixCallbackOp<FlagValuePairT, ScanBySegmentOp<ScanOp>>;

            Callable ZipValueAndFlags = [](UInt num_remaining,
                                           const ArrayVar<ValueType, ITEMS_PER_THREAD>& values,
                                           ArrayVar<int, ITEMS_PER_THREAD>& segment_flags,
                                           ArrayVar<FlagValuePairT, ITEMS_PER_THREAD>& scan_items,
                                           Bool is_last_tile) noexcept
            {
                for(auto item = 0; item < ITEMS_PER_THREAD; item++)
                {
                    $if(is_last_tile & (thread_id().x * UInt(ITEMS_PER_THREAD)) + item == num_remaining)
                    {
                        segment_flags[item] = 1;
                    };
                    scan_items[item].key   = segment_flags[item];
                    scan_items[item].value = values[item];
                }
            };

            Callable UnzipValues = [](ArrayVar<ValueType, ITEMS_PER_THREAD>& values,
                                      const ArrayVar<FlagValuePairT, ITEMS_PER_THREAD>& scan_items) noexcept
            {
                for(auto item = 0; item < ITEMS_PER_THREAD; item++)
                {
                    values[item] = scan_items[item].value;
                }
            };

            Callable AddInitToScan = [&](ArrayVar<ValueType, ITEMS_PER_THREAD>& scan_values,
                                         const ArrayVar<int, ITEMS_PER_THREAD>& segment_flags,
                                         const Var<ValueType>&                  init_value) noexcept
            {
                for(auto item = 0; item < ITEMS_PER_THREAD; item++)
                {
                    $if(segment_flags[item] == 1)
                    {
                        scan_values[item] = init_value;
                    }
                    $else
                    {
                        scan_values[item] = scan_op(scan_values[item], init_value);
                    };
                }
            };

            U<ScanByKeyKernel> scan_by_key_shader = nullptr;
            lazy_compile(
                device,
                scan_by_key_shader,
                [&](BufferVar<ScanTileState> tile_state,
                    BufferVar<KeyType>       d_keys_in,
                    BufferVar<KeyType>       d_prev_keys_in,
                    BufferVar<ValueType>     d_values_in,
                    BufferVar<ValueType>     d_values_out,
                    Var<ValueType>           init_value,
                    UInt                     num_item) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt thid       = thread_id().x;
                    UInt tile_id    = block_id().x;
                    UInt tile_items = UInt(ITEMS_PER_THREAD) * block_size_x();
                    UInt tile_start = tile_id * tile_items;

                    UInt num_remaining = num_item - tile_start;
                    Bool is_last_tile  = num_remaining <= tile_items;

                    SmemTypePtr<KeyType>   s_keys   = new SmemType<KeyType>{shared_mem_size};
                    SmemTypePtr<ValueType> s_values = new SmemType<ValueType>{shared_mem_size};

                    ArrayVar<KeyType, ITEMS_PER_THREAD>        local_keys;
                    ArrayVar<ValueType, ITEMS_PER_THREAD>      local_values;
                    ArrayVar<int, ITEMS_PER_THREAD>            local_segment_flags;
                    ArrayVar<FlagValuePairT, ITEMS_PER_THREAD> local_scan_items;

                    $if(is_last_tile)
                    {
                        BlockLoad<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_keys).Load(
                            d_keys_in, local_keys, tile_start, num_item - tile_start);
                        BlockLoad<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values).Load(
                            d_values_in, local_values, tile_start, num_item - tile_start);
                    }
                    $else
                    {
                        BlockLoad<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_keys).Load(d_keys_in, local_keys, tile_start);
                        BlockLoad<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values).Load(
                            d_values_in, local_values, tile_start);
                    };

                    sync_block();

                    ArrayVar<FlagValuePairT, ITEMS_PER_THREAD> output_scan_items;
                    ArrayVar<KeyType, ITEMS_PER_THREAD>        local_prev_keys;
                    Var<FlagValuePairT>                        tile_aggregate;
                    $if(tile_id == 0)
                    {
                        BlockDiscontinuity<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>().FlagHeads(
                            local_segment_flags,
                            local_prev_keys,
                            local_keys,
                            [](const Var<KeyType>& a, const Var<KeyType>& b) { return a != b; });

                        ZipValueAndFlags(num_remaining, local_values, local_segment_flags, local_scan_items, is_last_tile);

                        if constexpr(is_inclusive)
                        {
                            BlockScan<FlagValuePairT, BLOCK_SIZE, ITEMS_PER_THREAD>().InclusiveScan(
                                local_scan_items, output_scan_items, tile_aggregate, pair_scan_op);
                        }
                        else
                        {
                            BlockScan<FlagValuePairT, BLOCK_SIZE, ITEMS_PER_THREAD>().ExclusiveScan(
                                local_scan_items, output_scan_items, tile_aggregate, pair_scan_op);
                        }
                        $if(thread_id().x == 0)
                        {
                            $if(!is_last_tile)
                            {
                                // first tile
                                ScanTileStateViewer::SetInclusive(tile_state, 0, tile_aggregate);
                            };
                            output_scan_items[0].key = 0;
                        };
                    }
                    $else
                    {
                        Var<KeyType> tile_pred_key =
                            select(KeyType(0), d_prev_keys_in.read(tile_id), thread_id().x == 0);
                        BlockDiscontinuity<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>().FlagHeads(
                            local_segment_flags,
                            local_prev_keys,
                            local_keys,
                            [](const Var<KeyType>& a, const Var<KeyType>& b) { return a != b; },
                            tile_pred_key);

                        ZipValueAndFlags(num_remaining, local_values, local_segment_flags, local_scan_items, is_last_tile);

                        auto temp_storage = new SmemType<TilePrefixTempStorage<FlagValuePairT>>{1};
                        TilePrefixOpT prefix_op(tile_state, temp_storage, pair_scan_op, tile_id);
                        if constexpr(is_inclusive)
                        {
                            BlockScan<FlagValuePairT, BLOCK_SIZE, ITEMS_PER_THREAD>().InclusiveScan(
                                local_scan_items, output_scan_items, pair_scan_op, prefix_op);
                            tile_aggregate = prefix_op.GetBlockAggregate();
                        }
                        else
                        {
                            BlockScan<FlagValuePairT, BLOCK_SIZE, ITEMS_PER_THREAD>().ExclusiveScan(
                                local_scan_items, output_scan_items, pair_scan_op, prefix_op);
                            tile_aggregate = prefix_op.GetBlockAggregate();
                        }
                        device_log("thid: {},tile_pred_key:{}, output_scan_items: {},local_segment_flags:{}",
                                   dispatch_id().x,
                                   tile_pred_key,
                                   output_scan_items,
                                   local_segment_flags);
                    };

                    sync_block();

                    ArrayVar<ValueType, ITEMS_PER_THREAD> value_output;
                    UnzipValues(value_output, output_scan_items);

                    AddInitToScan(value_output, local_segment_flags, init_value);

                    $if(is_last_tile)
                    {
                        BlockStore<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values).Store(
                            value_output, d_values_out, tile_start, num_remaining);
                    }
                    $else
                    {
                        BlockStore<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values).Store(
                            value_output, d_values_out, tile_start);
                    };
                });

            return scan_by_key_shader;
        }
    };
};  // namespace details
};  // namespace luisa::parallel_primitive