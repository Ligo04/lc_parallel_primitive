/*
 * @Author: Ligo 
 * @Date: 2025-11-07 14:53:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 16:56:15
 */


#pragma once

#include "luisa/core/basic_traits.h"
#include "luisa/dsl/struct.h"
#include "luisa/dsl/sugar.h"
#include "luisa/dsl/var.h"
namespace luisa::parallel_primitive
{
struct GridEvenShared
{
  public:
    int           total_tiles;
    int           big_shares;
    compute::uint big_shared_items;
    compute::uint normal_share_items;
    compute::uint normal_base_offset;

    compute::uint num_items;
    int           grid_size;
    compute::uint block_offset;
    compute::uint block_end;
    compute::uint block_stride;
};
}  // namespace luisa::parallel_primitive

// #define LUISA_GRADEVENSHARED_TEMPLATE() template <NumericT Type4Byte>
// #define LUISA_GRADEVENSHARED_NAME() luisa::parallel_primitive::GridEvenShared

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
