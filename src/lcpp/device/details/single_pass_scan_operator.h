/*
 * @Author: Ligo 
 * @Date: 2025-10-22 11:24:49 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 15:23:01
 */
#pragma once
#include "lcpp/warp/warp_reduce.h"
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/stmt.h"
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

template <typename StatusWord, typename T>
struct TileDescriptor
{
    StatusWord status;
    T          value;

    TileDescriptor()
        : status(static_cast<StatusWord>(ScanTileStatus::SCAN_TILE_INVALID))
        , value(T(0)) {};

    TileDescriptor(const TileDescriptor&)            = default;
    TileDescriptor& operator=(const TileDescriptor&) = default;
};


template <typename T, bool SINGLE_WORD = is_numeric_v<T>>
struct ScanTileState;
// device and host
template <typename T>
struct ScanTileState<T, true>
{
    static_assert(sizeof(T) == 8 || sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1,
                  "Unsupported type size for ScanTileState");

    using StatusWord =
        std::conditional_t<sizeof(T) == 8, ulong, std::conditional_t<sizeof(T) == 4, uint, std::conditional_t<sizeof(T) == 2, ushort, uchar>>>;

    using TileDescriptorT = TileDescriptor<StatusWord, T>;
    TileDescriptorT d_tile_descriptions;
};

template <typename T>
struct ScanTileState<T, false>
{
    using StatusWord = uint;

    using TileDescriptorT = TileDescriptor<StatusWord, T>;

    static constexpr size_t description_bytes_per_tile = sizeof(StatusWord);
    static constexpr size_t payload_bytes_per_tile     = sizeof(T);

    StatusWord d_tile_status;
    T          d_tile_partial;
    T          d_tile_inclusive;
};

template <typename T, bool SINGLE_WORD = is_numeric_v<T>>
struct ScanTileStateViewer;
template <typename T>
struct ScanTileStateViewer<T, true>
{

    using StatusWordT     = typename ScanTileState<T, true>::StatusWord;
    using TileDescriptorT = typename ScanTileState<T, true>::TileDescriptorT;

    constexpr static size_t TILE_STATUS_PADDING = details::WARP_SIZE;

    static void InitializeWardStatus(compute::BufferVar<ScanTileState<T, true>>& tile_state,
                                     compute::UInt num_tile) noexcept
    {

        compute::UInt tile_idx = compute::dispatch_id().x;

        // compute::Var<TileDescriptorT>        tile_descriptor;
        compute::Var<ScanTileState<T, true>> state;

        $if(tile_idx < num_tile)
        {
            state.d_tile_descriptions.status =
                compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INVALID));
            state.d_tile_descriptions.value = T(0);
            tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_idx, state);
        };
        $if(compute::block_id().x == 0 & compute::thread_x() < compute::UInt(TILE_STATUS_PADDING))
        {
            state.d_tile_descriptions.status =
                compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_OBB));
            state.d_tile_descriptions.value = T(0);
            tile_state.write(compute::thread_x(), state);
        };
    };

    static void SetInclusive(compute::BufferVar<ScanTileState<T, true>>& tile_state,
                             compute::UInt          tile_index,
                             const compute::Var<T>& tile_prefix) noexcept
    {
        compute::Var<ScanTileState<T, true>> state;
        state.d_tile_descriptions.status = StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);
        state.d_tile_descriptions.value = tile_prefix;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    static void SetPartial(compute::BufferVar<ScanTileState<T, true>>& tile_state,
                           compute::UInt          tile_index,
                           const compute::Var<T>& tile_partial) noexcept
    {
        compute::Var<ScanTileState<T, true>> state;
        state.d_tile_descriptions.status = StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL);
        state.d_tile_descriptions.value = tile_partial;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    template <typename DelayT>
    static void WaitForValid(compute::BufferVar<ScanTileState<T>>& tile_state,
                             compute::Int                          tile_index,
                             compute::Var<StatusWordT>&            out_status,
                             compute::Var<T>&                      out_value,
                             DelayT delay) noexcept
    {
        compute::Var<ScanTileState<T>> curr_tile_state;
        curr_tile_state =
            tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while(compute::warp_active_any(curr_tile_state.d_tile_descriptions.status
                                        == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID)))
        {
            curr_tile_state =
                tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };

        out_status = curr_tile_state.d_tile_descriptions.status;
        out_value  = curr_tile_state.d_tile_descriptions.value;
    };

    static compute::Var<T> LoadValid(compute::BufferVar<ScanTileState<T, true>>& tile_state,
                                     compute::Int tile_index,
                                     auto         delay) noexcept
    {
        auto tile_descriptor =
            tile_state.read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        return tile_descriptor.value;
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
template <typename T, typename ScanOpT, typename ScanTileStateT, typename DelayConstructorT = no_delay_constructor<T>>
class TilePrefixCallbackOp : public LuisaModule
{
  public:
    using WarpReduceT = WarpReduce<T, 32>;

    using StatusWordT = typename ScanTileStateT::StatusWord;

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

            ScanTileStateViewer<T>::SetPartial(tile_status, tile_index, block_aggregate);
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
        $while(compute::warp_active_all(
            predecessor_status != StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE)))
        {
            predecessor_idx -= compute::Int(details::WARP_SIZE);
            process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay());
            exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
        };

        $if(compute::thread_x() == 0)
        {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            ScanTileStateViewer<T>::SetInclusive(tile_status, tile_index, inclusive_prefix);
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
    void process_windows(compute::Int      predecessor_idx,
                         Var<StatusWordT>& predecessor_status,
                         Var<T>&           windows_aggregate,
                         DeLayT            delay)
    {
        Var<T> value;
        ScanTileStateViewer<T>::WaitForValid(
            tile_status, predecessor_idx, predecessor_status, value, delay);

        compute::Int tail_flag =
            predecessor_status == StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);

        windows_aggregate = WarpReduceT().TailSegmentedReduce(
            value, tail_flag, SwizzleScanOp<ScanOpT>(scan_op));
    }
};

}  // namespace luisa::parallel_primitive

#define LUISA_TILEDESCRIPTOR_TEMPLATE()                                        \
    template <typename StatusWord, typename U>
#define LUISA_TILEDESCRIPTOR_NAME()                                            \
    luisa::parallel_primitive::TileDescriptor<StatusWord, U>
LUISA_TEMPLATE_STRUCT(LUISA_TILEDESCRIPTOR_TEMPLATE, LUISA_TILEDESCRIPTOR_NAME, status, value){};


#define LUISA_T_TEMPLATE() template <typename U>

#define LUISA_SCANTILESTATE_TRUE_NAME()                                        \
    luisa::parallel_primitive::ScanTileState<U, true>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_SCANTILESTATE_TRUE_NAME, d_tile_descriptions){};

#define LUISA_SCANTILESTATE_FALSE_NAME()                                       \
    luisa::parallel_primitive::ScanTileState<U, false>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE,
                      LUISA_SCANTILESTATE_FALSE_NAME,
                      d_tile_status,
                      d_tile_partial,
                      d_tile_inclusive){void InitializeWardStatus() noexcept {}};

#define LUISA_TILEPREFIXTEMPSTORAGE_NAME()                                     \
    luisa::parallel_primitive::TilePrefixTempStorage<U>

LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE,
                      LUISA_TILEPREFIXTEMPSTORAGE_NAME,
                      exclusive_prefix,
                      inclusive_prefix,
                      block_aggregate){};