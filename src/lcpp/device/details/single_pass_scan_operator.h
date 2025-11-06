/*
 * @Author: Ligo 
 * @Date: 2025-10-22 11:24:49 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 15:23:01
 */
#pragma once
#include "lcpp/common/thread_operators.h"
#include "lcpp/warp/warp_reduce.h"
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/struct.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/runtime/core.h>


namespace luisa::parallel_primitive
{

enum class ScanTileStatus : uint
{
    SCAN_TILE_OBB,           // out-of-bounds
    SCAN_TILE_INVALID = 99,  // not yet valid
    SCAN_TILE_PARTIAL,       // tile aggregate is available
    SCAN_TILE_INCLUSIVE,     // inclusive tile prefix is available
};

template <typename T>
struct no_delay_constructor
{
    no_delay_constructor(compute::UInt) noexcept {};

    struct delay_t
    {
        void operator()() const noexcept {};
    };

    [[nodiscard]] delay_t operator()() const noexcept { return delay_t{}; };
};


template <typename T>
struct ScanTileState
{
    compute::uint status;
    T             value;
};


struct ScanTileStateViewer
{
    using StatusWordT = compute::uint;

    constexpr static size_t TILE_STATUS_PADDING = details::WARP_SIZE;

    template <typename T>
    static void InitializeWardStatus(compute::BufferVar<ScanTileState<T>>& tile_state, compute::UInt num_tile) noexcept
    {

        compute::UInt tile_idx = compute::dispatch_id().x;

        compute::Var<ScanTileState<T>> state;

        $if(tile_idx < num_tile)
        {
            state.status = compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INVALID));
            state.value  = T(0);
            tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_idx, state);
        };
        $if(compute::block_id().x == 0 & compute::thread_x() < compute::UInt(TILE_STATUS_PADDING))
        {
            state.status = compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_OBB));
            state.value  = T(0);
            tile_state.write(compute::thread_x(), state);
        };
    };

    template <NumericT KeyType, NumericT ValueType>
    static void InitializeWardStatus(compute::BufferVar<ScanTileState<KeyValuePair<KeyType, ValueType>>>& tile_state,
                                     compute::UInt num_tile) noexcept
    {

        compute::UInt tile_idx = compute::dispatch_id().x;

        compute::Var<ScanTileState<KeyValuePair<KeyType, ValueType>>> state;

        $if(tile_idx < num_tile)
        {
            state.status = compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INVALID));
            state.value  = {KeyType(0), ValueType(0)};
            tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_idx, state);
        };
        $if(compute::block_id().x == 0 & compute::thread_x() < compute::UInt(TILE_STATUS_PADDING))
        {
            state.status = compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_OBB));
            state.value  = {KeyType(0), ValueType(0)};
            tile_state.write(compute::thread_x(), state);
        };
    };

    template <typename T>
    static void SetInclusive(compute::BufferVar<ScanTileState<T>>& tile_state,
                             compute::UInt                         tile_index,
                             const compute::Var<T>&                tile_prefix) noexcept
    {
        compute::Var<ScanTileState<T>> state;
        state.status = StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);
        state.value  = tile_prefix;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };


    template <NumericT KeyType, NumericT ValueType>
    static void SetInclusive(compute::BufferVar<ScanTileState<KeyValuePair<KeyType, ValueType>>>& tile_state,
                             compute::UInt tile_index,
                             const compute::Var<KeyValuePair<KeyType, ValueType>>& tile_prefix) noexcept
    {
        compute::Var<ScanTileState<KeyValuePair<KeyType, ValueType>>> state;
        state.status      = StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);
        state.value.key   = tile_prefix.key;
        state.value.value = tile_prefix.value;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    template <typename T>
    static void SetPartial(compute::BufferVar<ScanTileState<T>>& tile_state,
                           compute::UInt                         tile_index,
                           const compute::Var<T>&                tile_partial) noexcept
    {
        compute::Var<ScanTileState<T>> state;
        state.status = StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL);
        state.value  = tile_partial;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    template <NumericT KeyType, NumericT ValueType>
    static void SetPartial(compute::BufferVar<ScanTileState<KeyValuePair<KeyType, ValueType>>>& tile_state,
                           compute::UInt tile_index,
                           const compute::Var<KeyValuePair<KeyType, ValueType>>& tile_partial) noexcept
    {
        compute::Var<ScanTileState<KeyValuePair<KeyType, ValueType>>> state;
        state.status      = StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL);
        state.value.key   = tile_partial.key;
        state.value.value = tile_partial.value;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    template <typename T, typename DelayT>
    static void WaitForValid(compute::BufferVar<ScanTileState<T>>& tile_state,
                             compute::Int                          tile_index,
                             compute::Var<StatusWordT>&            out_status,
                             compute::Var<T>&                      out_value,
                             DelayT                                delay) noexcept
    {
        compute::Var<ScanTileState<T>> curr_tile_state;
        curr_tile_state = tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while(compute::warp_active_any(curr_tile_state.status == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)))
        {
            curr_tile_state = tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };

        out_status = curr_tile_state.status;
        out_value  = curr_tile_state.value;
    };

    template <NumericT KeyType, NumericT ValueType, typename DelayT>
    static void WaitForValid(compute::BufferVar<ScanTileState<KeyValuePair<KeyType, ValueType>>>& tile_state,
                             compute::Int                                    tile_index,
                             compute::Var<StatusWordT>&                      out_status,
                             compute::Var<KeyValuePair<KeyType, ValueType>>& out_value,
                             DelayT                                          delay) noexcept
    {
        compute::Var<ScanTileState<KeyValuePair<KeyType, ValueType>>> curr_tile_state;
        curr_tile_state = tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while(compute::warp_active_any(curr_tile_state.status == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)))
        {
            curr_tile_state = tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };

        out_status      = curr_tile_state.status;
        out_value.key   = curr_tile_state.value.key;
        out_value.value = curr_tile_state.value.value;
    };

    template <typename T>
    static compute::Var<T> LoadValid(compute::BufferVar<ScanTileState<T>>& tile_state,
                                     compute::Int                          tile_index,
                                     auto                                  delay) noexcept
    {
        auto state = tile_state.read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        return state.value;
    };
};

template <typename T>
struct TilePrefixTempStorage
{
    T exclusive_prefix;
    T inclusive_prefix;
    T block_aggregate;
};

// Decoupled look-back(warp)
// only device
template <typename T, typename ScanOpT, typename ScanTileStateT = ScanTileState<T>, typename DelayConstructorT = no_delay_constructor<T>>
class TilePrefixCallbackOp : public LuisaModule
{
  public:
    using WarpReduceT = WarpReduce<T, details::WARP_SIZE>;

    using StatusWordT = compute::uint;

    using TempStorageT = TilePrefixTempStorage<T>;

    // TempStorageT&                        temp_storage;
    SmemTypePtr<TempStorageT>           temp_storage;
    compute::BufferVar<ScanTileStateT>& tile_status;
    ScanOpT                             scan_op;
    compute::Int                        tile_index;
    Var<T>                              exclusive_prefix;
    Var<T>                              inclusive_prefix;

    TilePrefixCallbackOp(compute::BufferVar<ScanTileStateT>& tile_state,
                         SmemTypePtr<TempStorageT>&          temp_storage,
                         ScanOpT                             scan_op,
                         compute::Int                        tile_index)
        : tile_status{tile_state}
        , temp_storage{temp_storage}
        , scan_op{scan_op}
        , tile_index{tile_index} {};

    TilePrefixCallbackOp(compute::BufferVar<ScanTileStateT>& tile_state,
                         SmemTypePtr<TempStorageT>&          temp_storage,
                         ScanOpT                             scan_op)
        : TilePrefixCallbackOp(tile_state, temp_storage, scan_op, compute::block_x()) {};

  public:
    Var<T> operator()(const Var<T>& block_aggregate)
    {
        $if(compute::thread_x() == 0)
        {
            (*temp_storage)[0].block_aggregate = block_aggregate;
            ScanTileStateViewer::SetPartial(tile_status, tile_index, block_aggregate);
        };

        compute::Int     predecessor_idx = tile_index - compute::thread_x() - 1;
        Var<StatusWordT> predecessor_status;
        Var<T>           windows_aggregate;

        // decay
        DelayConstructorT construct_delay(tile_index);
        process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay());

        // The exclusive tile prefix starts out as the current window aggregate
        exclusive_prefix = windows_aggregate;

        // warp(32) polling for predecessor tiles
        $while(compute::warp_active_all(predecessor_status != StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE)))
        {
            predecessor_idx -= compute::Int(details::WARP_SIZE);
            process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay());
            exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
        };

        $if(compute::thread_x() == 0)
        {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            ScanTileStateViewer::SetInclusive(tile_status, tile_index, inclusive_prefix);
            (*temp_storage)[0].exclusive_prefix = exclusive_prefix;
            (*temp_storage)[0].inclusive_prefix = inclusive_prefix;
        };

        return exclusive_prefix;
    }

    inline compute::UInt GetTileIndex() const noexcept { return tile_index; }

    inline compute::Var<T> GetInclusivePrefix() const noexcept
    {
        return (*temp_storage)[0].inclusive_prefix;
    };

    inline compute::Var<T> GetExclusivePrefix() const noexcept
    {
        return (*temp_storage)[0].exclusive_prefix;
    };

    inline compute::Var<T> GetBlockAggregate() const noexcept
    {
        return (*temp_storage)[0].block_aggregate;
    };

  private:
    template <typename DeLayT>
    void process_windows(compute::Int predecessor_idx, Var<StatusWordT>& predecessor_status, Var<T>& windows_aggregate, DeLayT delay)
    {
        Var<T> value;
        ScanTileStateViewer::WaitForValid(tile_status, predecessor_idx, predecessor_status, value, delay);

        compute::Int tail_flag = predecessor_status == StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);

        windows_aggregate =
            WarpReduceT().TailSegmentedReduce(value, tail_flag, SwizzleScanOp<ScanOpT>(scan_op));

        // compute::device_log("process_windows: tile_index: {}, status: {}, value: {}, aggregate: {}",
        //                     tile_index,
        //                     predecessor_status,
        //                     value,
        //                     windows_aggregate);
    }
};

}  // namespace luisa::parallel_primitive

#define LUISA_T_TEMPLATE() template <typename U>

#define LUISA_SCANTILESTATE_TRUE_NAME() luisa::parallel_primitive::ScanTileState<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_SCANTILESTATE_TRUE_NAME, status, value){};


#define LUISA_TILEPREFIXTEMPSTORAGE_NAME() luisa::parallel_primitive::TilePrefixTempStorage<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_TILEPREFIXTEMPSTORAGE_NAME, exclusive_prefix, inclusive_prefix, block_aggregate){};