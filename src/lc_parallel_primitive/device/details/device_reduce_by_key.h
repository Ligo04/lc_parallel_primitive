/*
 * @Author: Ligo 
 * @Date: 2025-10-21 22:12:15 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-21 23:16:45
 */

#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>
#include <lc_parallel_primitive/runtime/core.h>

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
            auto scatter =
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
                    // two phase scatter
                    // only ITEMS_PER_THREAD > 1 and num_tile_segment > BLOCK_SIZE need two phase scatter

                    sync_block();
                    // phase 1: write to shared memory
                }
                $else
                {
                    // direct scatter
                    $for(item, 0u, UInt(ITEMS_PER_THREAD))
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
                    UInt tile_start = block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);

                    SmemTypePtr<KeyType> s_keys = new SmemType<KeyType>{shared_mem_size};
                    SmemTypePtr<ValueType> s_values =
                        new SmemType<ValueType>{shared_mem_size};
                    SmemTypePtr<int> s_discontinutity = new SmemType<int>{shared_mem_size};

                    ArrayVar<KeyType, ITEMS_PER_THREAD>   local_keys;
                    ArrayVar<KeyType, ITEMS_PER_THREAD>   local_prev_keys;
                    ArrayVar<ValueType, ITEMS_PER_THREAD> local_values;
                    ArrayVar<int, ITEMS_PER_THREAD>       local_flags;

                    Bool is_last_tile = block_id().x == (block_size_x() - 1u);

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
                    $for(item, 0u, UInt(ITEMS_PER_THREAD))
                    {
                        scan_items[item] = {local_flags[item], local_values[item]};
                    };

                    ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scan_output;
                    Var<KeyValuePair<int, ValueType>> block_aggregate{0, 0};

                    BlockScan<KeyValuePair<int, ValueType>, BLOCK_SIZE, ITEMS_PER_THREAD>()
                        .ExclusiveScan(scan_items,
                                       scan_output,
                                       block_aggregate,
                                       ReduceBySegmentOp<ReduceOp>());

                    // device_log("thid {},key {},value {},flags {}, scan_output_key {}, scan_output_value {}, block_aggregate key {}, value {}",
                    //            block_id().x * block_size().x * UInt(ITEMS_PER_THREAD) + thid,
                    //            local_keys[0],
                    //            local_values[0],
                    //            local_flags[0],
                    //            scan_output[0].key,
                    //            scan_output[0].value,
                    //            block_aggregate.key,
                    //            block_aggregate.value);

                    ArrayVar<KeyValuePair<int, ValueType>, ITEMS_PER_THREAD> scatter_items;
                    ArrayVar<uint, ITEMS_PER_THREAD> scatter_indices;
                    $for(item, 0u, UInt(ITEMS_PER_THREAD))
                    {
                        scatter_items[item].key   = local_prev_keys[item];
                        scatter_items[item].value = scan_output[item].value;
                        scatter_indices[item]     = scan_output[item].key;
                    };

                    Int num_tile_segments = block_aggregate.key;
                    scatter(scatter_items, local_flags, scatter_indices, unique_out, aggregated_out, num_tile_segments, 0);
                });

            return ms_reduce_by_key_shader;
        }
    };

};  // namespace details
};  // namespace luisa::parallel_primitive