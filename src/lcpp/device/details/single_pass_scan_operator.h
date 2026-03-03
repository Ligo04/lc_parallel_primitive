/*
 * @Author: Ligo 
 * @Date: 2025-10-22 11:24:49 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:09:45
 */
#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/core/basic_traits.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/struct.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{

enum class ScanTileStatus : uint
{
    SCAN_TILE_OBB,           // out-of-bounds
    SCAN_TILE_INVALID = 99,  // not yet valid
    SCAN_TILE_PARTIAL,       // tile aggregate is available
    SCAN_TILE_INCLUSIVE,     // inclusive tile prefix is available
};

enum class MemoryOrder
{
    relaxed,
    acquire_release,
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

template <typename T>
struct ScanTileStateViewer
{

    using StatusWordT = compute::uint;

    constexpr static size_t TILE_STATUS_PADDING = details::WARP_SIZE;

    compute::BufferVar<compute::uint>& d_tile_status;
    compute::BufferVar<T>&             d_tile_partial;
    compute::BufferVar<T>&             d_tile_inclusive;

    ScanTileStateViewer(compute::BufferVar<StatusWordT>& tile_status,
                        compute::BufferVar<T>&           tile_partial,
                        compute::BufferVar<T>&           tile_inclusive)
        : d_tile_status(tile_status)
        , d_tile_partial(tile_partial)
        , d_tile_inclusive(tile_inclusive) {};



    void SetInclusive(compute::Int tile_index, const compute::Var<T>& tile_inclusive) noexcept
    {
        d_tile_inclusive.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index, tile_inclusive);
        d_tile_status.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index,
                                     compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE)));
    };

    void SetPartial(compute::Int tile_index, const compute::Var<T>& tile_partial) noexcept
    {
        d_tile_partial.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index, tile_partial);
        d_tile_status.volatile_write(compute::Int(TILE_STATUS_PADDING) + tile_index,
                                     compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL)));
    };

    template <typename DelayT>
    void WaitForValid(compute::Int tile_index, compute::Var<StatusWordT>& out_status, compute::Var<T>& out_value, DelayT delay) noexcept
    {
        out_status = d_tile_status.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while(compute::warp_active_any(out_status == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)))
        {
            delay();
            out_status = d_tile_status.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };
        $if(out_status == StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL))
        {
            out_value = d_tile_partial.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        }
        $else
        {
            out_value = d_tile_inclusive.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };
    };
};
static void InitializeWardStatus(compute::UInt num_tile, compute::BufferVar<compute::uint>& d_tile_status) noexcept
{
    compute::UInt tile_idx = compute::dispatch_id().x;
    $if(tile_idx < num_tile)
    {
        d_tile_status.write(compute::UInt(ScanTileStateViewer<int>::TILE_STATUS_PADDING) + tile_idx,
                            compute::def(ScanTileStateViewer<int>::StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)));
    };
    $if(compute::block_id().x == 0 & compute::thread_x() < compute::UInt(ScanTileStateViewer<int>::TILE_STATUS_PADDING))
    {
        d_tile_status.write(compute::thread_x(), compute::def(ScanTileStateViewer<int>::StatusWordT(ScanTileStatus::SCAN_TILE_OBB)));
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
template <typename T, typename ScanOpT, typename ScanTileStateT = ScanTileStateViewer<T>, typename DelayConstructorT = no_delay_constructor<T>>
class TilePrefixCallbackOp : public LuisaModule
{
  public:
    using WarpReduceT = WarpReduce<T, details::WARP_SIZE>;

    using StatusWordT = compute::uint;

    using TempStorageT = TilePrefixTempStorage<T>;

    // TempStorageT&                        temp_storage;
    SmemTypePtr<TempStorageT> temp_storage;
    // compute::BufferVar<ScanTileStateT>& tile_status;
    ScanTileStateT& tile_status;
    ScanOpT         scan_op;
    compute::UInt   tile_index;
    Var<T>          exclusive_prefix;
    Var<T>          inclusive_prefix;

    TilePrefixCallbackOp(ScanTileStateT&            tile_state,
                         SmemTypePtr<TempStorageT>& temp_storage,
                         ScanOpT                    scan_op,
                         compute::UInt              tile_index)
        : tile_status{tile_state}
        , temp_storage{temp_storage}
        , scan_op{scan_op}
        , tile_index{tile_index} {};

    TilePrefixCallbackOp(ScanTileStateT& tile_state, SmemTypePtr<TempStorageT>& temp_storage, ScanOpT scan_op)
        : TilePrefixCallbackOp(tile_state, temp_storage, scan_op, compute::block_x()) {};

  public:
    Var<T> operator()(const Var<T>& block_aggregate)
    {
        $if(compute::thread_x() == 0)
        {
            (*temp_storage)[0].block_aggregate = block_aggregate;
            // ScanTileStateViewer::SetPartial(tile_status, tile_index, block_aggregate);
            tile_status.SetPartial(tile_index, block_aggregate);
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
            // ScanTileStateViewer::SetInclusive(tile_status, tile_index, inclusive_prefix);
            tile_status.SetInclusive(tile_index, inclusive_prefix);
            (*temp_storage)[0].exclusive_prefix = exclusive_prefix;
            (*temp_storage)[0].inclusive_prefix = inclusive_prefix;
            // device_log("Tile {}: exclusive = {} inclusive = {} ", tile_index, exclusive_prefix, inclusive_prefix);
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
        // ScanTileStateViewer::WaitForValid(tile_status, predecessor_idx, predecessor_status, value, delay);
        tile_status.WaitForValid(predecessor_idx, predecessor_status, value, delay);

        compute::UInt tail_flag = (predecessor_status == StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE));
        windows_aggregate =
            WarpReduceT().TailSegmentedReduce(value, tail_flag, SwizzleScanOp<ScanOpT>(scan_op));
        // $if(tail_flag == 0 & value != 1 & value != 0)
        // {
        //     device_log("Tile {}: process_windows predecessor_idx = {}, predecessor_status = {}, tail_flag = {}, value = {}, windows_aggregate = {}",
        //                tile_index,
        //                predecessor_idx,
        //                predecessor_status,
        //                tail_flag,
        //                value,
        //                windows_aggregate);
        // };
    }
};

}  // namespace luisa::parallel_primitive

#define LUISA_T_TEMPLATE() template <typename U>

#define LUISA_SCANTILESTATE_TRUE_NAME() luisa::parallel_primitive::ScanTileState<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_SCANTILESTATE_TRUE_NAME, status, value){};


#define LUISA_TILEPREFIXTEMPSTORAGE_NAME() luisa::parallel_primitive::TilePrefixTempStorage<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_TILEPREFIXTEMPSTORAGE_NAME, exclusive_prefix, inclusive_prefix, block_aggregate){};