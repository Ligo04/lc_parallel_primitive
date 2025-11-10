/*
 * @Author: Ligo 
 * @Date: 2025-10-21 23:03:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 16:33:11
 */

#pragma once
#include "luisa/dsl/stmt.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/device/details/single_pass_scan_operator.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
    class ScanModule : public LuisaModule
    {
      public:
        using TileState = ScanTileState<Type4Byte>;

        using ScanTileStateInitKernel = Shader<1, Buffer<TileState>, int>;

        using ScanKernel = Shader<1, Buffer<TileState>, Buffer<Type4Byte>, Buffer<Type4Byte>, Type4Byte, uint>;

        template <typename ScanOP>
        using TilePrefixOpT = TilePrefixCallbackOp<Type4Byte, ScanOP>;

        U<ScanTileStateInitKernel> compile_scan_tile_state_init(Device& device)
        {
            U<ScanTileStateInitKernel> scan_tile_state_init_shader = nullptr;
            lazy_compile(device,
                         scan_tile_state_init_shader,
                         [](BufferVar<TileState> tile_state, Int num_tiles) noexcept
                         { ScanTileStateViewer::InitializeWardStatus(tile_state, num_tiles); });
            return scan_tile_state_init_shader;
        }

        template <bool is_inclusive, typename ScanOp>
        U<ScanKernel> compile(Device& device, size_t shared_mem_size, ScanOp scan_op)
        {
            U<ScanKernel> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<TileState> tile_state,
                    BufferVar<Type4Byte> d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                 num_elements)
                {
                    set_block_size(BLOCK_SIZE);
                    UInt thid       = thread_id().x;
                    UInt tile_id    = block_id().x;
                    UInt tile_items = UInt(ITEMS_PER_THREAD) * block_size_x();
                    UInt tile_start = tile_id * tile_items;

                    UInt num_remaining = num_elements - tile_start;
                    Bool is_last_tile  = num_remaining <= tile_items;

                    ArrayVar<Type4Byte, ITEMS_PER_THREAD> items;
                    SmemTypePtr<Type4Byte> s_data = new SmemType<Type4Byte>{shared_mem_size};
                    $if(is_last_tile)
                    {
                        BlockLoad<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>(s_data).Load(
                            d_in, items, tile_start, num_remaining);
                    }
                    $else
                    {
                        BlockLoad<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>(s_data).Load(d_in, items, tile_start);
                    };
                    sync_block();

                    ArrayVar<Type4Byte, ITEMS_PER_THREAD> output_items;
                    $if(tile_id == 0)
                    {
                        Var<Type4Byte>                                     block_aggregate;
                        BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                        if constexpr(is_inclusive)
                        {
                            block_scan.InclusiveScan(items, output_items, block_aggregate, scan_op, init_value);
                            block_aggregate = scan_op(block_aggregate, init_value);
                        }
                        else
                        {
                            block_scan.ExclusiveScan(items, output_items, block_aggregate, scan_op, init_value);
                            block_aggregate = scan_op(block_aggregate, init_value);
                        }
                        $if(!is_last_tile & thread_id().x == 0)
                        {
                            // first tile
                            ScanTileStateViewer::SetInclusive(tile_state, 0, block_aggregate);
                        };
                    }
                    $else
                    {
                        auto temp_storage = new SmemType<TilePrefixTempStorage<Type4Byte>>{1};
                        TilePrefixOpT prefix_op(tile_state, temp_storage, scan_op, tile_id);
                        BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                        if constexpr(is_inclusive)
                        {
                            block_scan.InclusiveScan(items, output_items, scan_op, prefix_op);
                        }
                        else
                        {
                            block_scan.ExclusiveScan(items, output_items, scan_op, prefix_op);
                        }
                    };
                    // device_log("thid: {}, thread_id: {}, items: {}, output_items: {}", dispatch_id().x, thid, items, output_items);

                    sync_block();
                    $if(is_last_tile)
                    {
                        BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>(s_data).Store(
                            output_items, d_out, tile_start, num_remaining);
                    }
                    $else
                    {
                        BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>(s_data).Store(output_items, d_out, tile_start);
                    };
                });

            return scan_shader;
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive