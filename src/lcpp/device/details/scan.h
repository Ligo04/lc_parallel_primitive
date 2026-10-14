/*
 * @Author: Ligo 
 * @Date: 2025-10-21 23:03:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 16:33:11
 */

#pragma once
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
        using TileStateViewer = ScanTileStateViewer<Type4Byte>;

        using ScanTileStateInitKernel = Shader<1, Buffer<uint>, Buffer<Type4Byte>, Buffer<Type4Byte>, uint>;

        using ScanKernel =
            Shader<1, Buffer<uint>, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, Type4Byte, uint>;

        template <typename ScanOP>
        using TilePrefixOpT = TilePrefixCallbackOp<Type4Byte, ScanOP>;

        U<ScanTileStateInitKernel> compile_scan_tile_state_init(Device& device)
        {
            U<ScanTileStateInitKernel> scan_tile_state_init_shader = nullptr;
            lazy_compile(device,
                         scan_tile_state_init_shader,
                         [](BufferVar<uint>      tile_status,
                            BufferVar<Type4Byte> tile_partial,
                            BufferVar<Type4Byte> tile_inclusive,
                            UInt                 num_tiles) noexcept
                         {
                             InitializeWardStatus(num_tiles, tile_status);
                         });
            return scan_tile_state_init_shader;
        }

        template <bool is_inclusive, typename ScanOp>
        U<ScanKernel> compile(Device& device, size_t shared_mem_size, ScanOp scan_op)
        {
            U<ScanKernel> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<uint>      tile_status,
                    BufferVar<Type4Byte> tile_partial,
                    BufferVar<Type4Byte> tile_inclusive,
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

                    TileStateViewer tile_state_viewer(tile_status, tile_partial, tile_inclusive);

                    ArrayVar<Type4Byte, ITEMS_PER_THREAD> items;
                    SmemTypePtr<Type4Byte> s_data = new SmemType<Type4Byte>{shared_mem_size};
                    // Coalesced (block-striped) global I/O transposed through shared
                    // memory so the per-thread register array stays in the blocked,
                    // thread-major order that BlockScan requires for an order-correct
                    // prefix sum. Thread `thid` fetches tile positions
                    // {i*BLOCK + thid} (adjacent threads -> adjacent addresses) into
                    // shared, then reads the blocked range {thid*ITEMS + i} back into
                    // registers. The end bound `num_remaining` is compared against the
                    // RELATIVE tile position before adding `tile_start`, exactly as the
                    // previous guarded DIRECT load did.
                    UInt block_stride = UInt(BLOCK_SIZE);
                    $if(is_last_tile)
                    {
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt p = UInt(i) * block_stride + thid; // coalesced tile position
                            $if(p < num_remaining)
                            {
                                (*s_data)[p] = d_in.read(tile_start + p);
                            };
                        };
                        sync_block();
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt l = thid * UInt(ITEMS_PER_THREAD) + UInt(i); // blocked logical position
                            $if(l < num_remaining)
                            {
                                items[i] = (*s_data)[l];
                            }
                            $else
                            {
                                items[i] = Type4Byte(0);
                            };
                        };
                    }
                    $else
                    {
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt p = UInt(i) * block_stride + thid;
                            (*s_data)[p] = d_in.read(tile_start + p);
                        };
                        sync_block();
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt l = thid * UInt(ITEMS_PER_THREAD) + UInt(i);
                            items[i] = (*s_data)[l];
                        };
                    };

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
                            tile_state_viewer.SetInclusive(0, block_aggregate);
                        };
                    }
                    $else
                    {
                        auto temp_storage = new SmemType<TilePrefixTempStorage<Type4Byte>>{1};
                        TilePrefixCallbackOp prefix_op(tile_state_viewer, temp_storage, scan_op, tile_id);
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

                    sync_block();
                    // Reverse transpose: write the scan results back into shared at
                    // their blocked logical positions, then store to global in the
                    // coalesced (block-striped) order. The (thid, i) -> l mapping here
                    // is identical to the load side, so element ordering is preserved.
                    for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                    {
                        UInt l = thid * UInt(ITEMS_PER_THREAD) + UInt(i);
                        (*s_data)[l] = output_items[i];
                    };
                    sync_block();
                    $if(is_last_tile)
                    {
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt p = UInt(i) * block_stride + thid;
                            $if(p < num_remaining)
                            {
                                d_out.write(tile_start + p, (*s_data)[p]);
                            };
                        };
                    }
                    $else
                    {
                        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
                        {
                            UInt p = UInt(i) * block_stride + thid;
                            d_out.write(tile_start + p, (*s_data)[p]);
                        };
                    };
                });

            return scan_shader;
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive