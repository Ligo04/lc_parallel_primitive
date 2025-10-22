/*
 * @Author: Ligo 
 * @Date: 2025-10-21 22:12:15 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:56:51
 */

#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/block/block_scan.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT KeyType, NumericT ValueType>
    using ReduceByKeyShaderT =
        Shader<1, Buffer<KeyType>, Buffer<ValueType>, Buffer<KeyType>, Buffer<ValueType>, Buffer<uint>, uint>;

    template <NumericT KeyType, NumericT ValueType, size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD>
    class ReduceByKeyShader : public LuisaModule
    {
      public:
        template <typename ReduceOp>
        luisa::unique_ptr<ReduceByKeyShaderT<KeyType, ValueType>> compile(
            Device& device, size_t shared_mem_size, ReduceOp reduce_op)
        {
            auto scatter_op =
                [&](const ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD>& scatter_items,
                    const ArrayVar<int, ITEMS_PER_THREAD>& segment_flags,
                    const ArrayVar<int, ITEMS_PER_THREAD>& segment_indices,
                    BufferVar<KeyType>&                    d_unique_out,
                    BufferVar<ValueType>&                  d_aggregated_out,
                    Int                                    num_tile_segment,
                    Int num_segments_prefix) noexcept
            {
                $if(UInt(ITEMS_PER_THREAD) > 1 & num_tile_segment > UInt(BLOCK_SIZE))
                {
                    // device_log("Two phase scatter is not implemented yet.");
                    // two phase scatter
                    // only ITEMS_PER_THREAD > 1 and num_tile_segment > BLOCK_SIZE need two phase scatter
                    SmemTypePtr<KeyValuePair<KeyType, ValueType>> s_scatter_keys =
                        new SmemType<KeyValuePair<KeyType, ValueType>>{shared_mem_size};
                    sync_block();
                    // phase 1: write to shared memory
                    $for(item, 0u, UInt(ITEMS_PER_THREAD))
                    {

                        $if(segment_flags[item] == 1)
                        {
                            (*s_scatter_keys)[segment_indices[item] - num_segments_prefix] =
                                scatter_items[item];
                        };
                    };

                    sync_block();
                    // phase 2: write to global memory
                    UInt item = thread_id().x;
                    $while(item < num_tile_segment)
                    {
                        Var<KeyValuePair<KeyType, ValueType>> kvp =
                            (*s_scatter_keys)[item];
                        d_unique_out.write(item + num_segments_prefix, kvp.key);
                        d_aggregated_out.write(item + num_segments_prefix, kvp.value);
                        item += UInt(block_size_x());
                    };
                }
                $else
                {
                    // direct scatter
                    // for(item, 0u, UInt(ITEMS_PER_THREAD))
                    for(auto item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        $if(segment_flags[item] == 1)
                        {
                            d_unique_out.write(segment_indices[item],
                                               scatter_items[item].key);
                            d_aggregated_out.write(segment_indices[item],
                                                   scatter_items[item].value);
                        };
                    };
                };
            };


            luisa::unique_ptr<ReduceByKeyShaderT<KeyType, ValueType>> ms_reduce_by_key_shader =
                nullptr;

            lazy_compile(
                device,
                ms_reduce_by_key_shader,
                [&](BufferVar<KeyType>   keys_in,
                    BufferVar<ValueType> values_in,
                    BufferVar<KeyType>   unique_out,
                    BufferVar<ValueType> aggregated_out,
                    BufferVar<uint>      num_runs_out,
                    UInt                 num_item) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt thid    = UInt(thread_id().x);
                    UInt tile_id = block_id().x;
                    UInt tile_items = UInt(ITEMS_PER_THREAD) * UInt(block_size_x());
                    UInt tile_start = tile_id * tile_items;

                    UInt num_tile_items = num_item - tile_start;
                    Bool is_last_tile   = num_tile_items <= tile_items;

                    SmemTypePtr<KeyType> s_keys = new SmemType<KeyType>{shared_mem_size};
                    SmemTypePtr<ValueType> s_values =
                        new SmemType<ValueType>{shared_mem_size};
                    SmemTypePtr<int> s_discontinutity = new SmemType<int>{shared_mem_size};

                    ArrayVar<KeyType, ITEMS_PER_THREAD>   local_keys;
                    ArrayVar<KeyType, ITEMS_PER_THREAD>   local_prev_keys;
                    ArrayVar<ValueType, ITEMS_PER_THREAD> local_values;
                    ArrayVar<int, ITEMS_PER_THREAD>       local_flags;


                    $if(is_last_tile)
                    {
                        BlockLoad<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_keys).Load(
                            keys_in, local_keys, tile_start, num_item - tile_start);
                        BlockLoad<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values)
                            .Load(values_in, local_values, tile_start, num_item - tile_start);
                    }
                    $else
                    {
                        BlockLoad<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_keys).Load(
                            keys_in, local_keys, tile_start);
                        BlockLoad<ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>(s_values)
                            .Load(values_in, local_values, tile_start);
                    };

                    // load per keys
                    Var<KeyType> tile_predecessor;
                    $if(thid == 0)
                    {
                        $if(tile_id == 0)
                        {
                            tile_predecessor = -1;
                        }
                        $else
                        {
                            tile_predecessor = keys_in.read(tile_start - 1);
                        };
                    };

                    BlockDiscontinuity<KeyType, BLOCK_SIZE, ITEMS_PER_THREAD>().FlagHeads(
                        local_flags,
                        local_prev_keys,
                        local_keys,
                        [&](const Var<KeyType>& a, const Var<KeyType>& b)
                        { return a != b; },
                        tile_predecessor);

                    $if(thid == 0 & tile_id == 0)
                    {
                        local_flags[0] = 0;
                    };

                    ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scan_items;
                    for(auto item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        scan_items[item] = {local_flags[item], local_values[item]};
                    };

                    Int                               num_segment_prefix = 0;
                    Var<KeyValuePair<int, ValueType>> total_aggreagate{0, 0};
                    ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scan_output;
                    Var<KeyValuePair<int, ValueType>> block_aggregate{0, 0};

                    $if(tile_id == 0)
                    {
                        BlockScan<KeyValuePair<int, ValueType>, BLOCK_SIZE, ITEMS_PER_THREAD>()
                            .ExclusiveScan(scan_items,
                                           scan_output,
                                           block_aggregate,
                                           ReduceBySegmentOp<ReduceOp>());
                        total_aggreagate   = block_aggregate;
                        num_segment_prefix = 0;
                    };


                    ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scatter_items;
                    ArrayVar<uint, ITEMS_PER_THREAD> scatter_indices;
                    for(auto item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        scatter_items[item].key   = local_prev_keys[item];
                        scatter_items[item].value = scan_output[item].value;
                        scatter_indices[item]     = scan_output[item].key;

                        // device_log("thid: {},item:{}, scan_output_key: {}, scan_output_value: {}, block_aggregate.key: {}, block_aggregate.value: {}",
                        //            thid,
                        //            item,
                        //            scan_output[item].key,
                        //            scan_output[item].value,
                        //            block_aggregate.key,
                        //            block_aggregate.value);
                    };

                    Int num_tile_segments = block_aggregate.key;
                    scatter_op(scatter_items,
                               local_flags,
                               scatter_indices,
                               unique_out,
                               aggregated_out,
                               num_tile_segments,
                               0);

                    // last tile and last thread
                    $if(is_last_tile & thread_id().x == block_size_x() - 1u)
                    {
                        UInt num_segments = num_tile_segments + num_segment_prefix;
                        $if(num_tile_items == tile_items)
                        {
                            unique_out.write(num_segments,
                                             local_keys[ITEMS_PER_THREAD - 1]);
                            aggregated_out.write(num_segments, total_aggreagate.value);
                            num_segments += 1;
                        };
                        num_runs_out.write(0, num_segments);
                    };
                });

            return ms_reduce_by_key_shader;
        }
    };

};  // namespace details
};  // namespace luisa::parallel_primitive