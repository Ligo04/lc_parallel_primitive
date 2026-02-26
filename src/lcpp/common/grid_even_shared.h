/*
 * @Author: Ligo 
 * @Date: 2025-11-07 14:53:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 16:56:15
 */


#pragma once

#include "lcpp/common/utils.h"
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/struct.h"
#include "luisa/dsl/sugar.h"
#include "luisa/dsl/var.h"
namespace luisa::parallel_primitive
{
struct GridEvenShared
{
  public:
    compute::uint total_tiles;
    compute::uint big_shares;
    compute::uint big_shared_items;
    compute::uint normal_share_items;
    compute::uint normal_base_offset;

    compute::uint num_items;
    compute::uint grid_size;
    compute::uint block_offset;
    compute::uint block_end;
    compute::uint block_stride;

    void DispatchInit(compute::uint num_items, compute::uint max_grid_size, compute::uint tile_items)
    {
        if(num_items <= 0 || max_grid_size <= 0 || tile_items <= 0)
        {
            // invalid dispatch, set grid size to 1 and let first block handle it
            this->num_items    = 0;
            this->grid_size    = 0;
            this->block_offset = 0;
            this->block_end    = 0;
            return;
        }

        this->block_offset = num_items;
        this->block_end    = num_items;
        this->num_items    = num_items;

        this->total_tiles = std::min(std::numeric_limits<uint>::max(), ceil_div(num_items, tile_items));
        this->grid_size = std::min(total_tiles, max_grid_size);

        compute::uint avg_items_per_tile = grid_size > 0 ? total_tiles / grid_size : 0;
        this->big_shares                 = total_tiles - (avg_items_per_tile * grid_size);
        this->normal_share_items         = avg_items_per_tile * tile_items;
        this->normal_base_offset         = big_shares * tile_items;
        this->big_shared_items           = this->normal_share_items + tile_items;
    };
};
}  // namespace luisa::parallel_primitive
LUISA_STRUCT(luisa::parallel_primitive::GridEvenShared,
             total_tiles,
             big_shares,
             big_shared_items,
             normal_share_items,
             normal_base_offset,
             num_items,
             grid_size,
             block_offset,
             block_end,
             block_stride)
{
  public:
    template <int TILE_ITEMS>
    void BlockInit(luisa::compute::UInt block_id)
    {
        this->block_stride = luisa::compute::UInt(TILE_ITEMS);
        $if(block_id < big_shares)
        {
            this->block_offset = block_id * big_shared_items;
            this->block_end    = this->block_offset + big_shared_items;
        }
        $elif(block_id < luisa::compute::UInt(total_tiles))
        {
            this->block_offset = normal_base_offset + (block_id * normal_share_items);
            this->block_end    = this->block_offset
                              + luisa::compute::min(num_items - this->block_offset, normal_share_items);
        };
    };

    template <int TILE_ITEMS>
    void BlockInit()
    {
        BlockInit<TILE_ITEMS>(luisa::compute::block_id().x);
    };

    template <int TILE_ITEMS>
    void BlockInit(luisa::compute::UInt block_offset, luisa::compute::UInt block_end)
    {
        this->block_offset = block_offset;
        this->block_end    = block_end;
        this->block_stride = luisa::compute::UInt(TILE_ITEMS);
    };
};
