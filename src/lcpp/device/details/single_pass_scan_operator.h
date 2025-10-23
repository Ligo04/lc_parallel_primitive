/*
 * @Author: Ligo 
 * @Date: 2025-10-22 11:24:49 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 15:23:01
 */
#pragma once
#include "lcpp/warp/warp_reduce.h"
#include "luisa/ast/op.h"
#include "luisa/core/basic_traits.h"
#include "luisa/core/mathematics.h"
#include "luisa/vstl/config.h"
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
using namespace luisa::compute;


enum class ScanTileStatus : uint
{
    SCAN_TILE_OBB,
    SCAN_TILE_INVALID,
    SCAN_TILE_PARTIAL,
    SCAN_TILE_INCLUSIVE,  // for inclusive scan
};

template <typename T, bool SINGLE_WORD = is_numeric_v<T>>
struct ScanTileState;

// For single word type
template <typename T>
struct ScanTileState<T, true>
{
    using StatusValueT = T;

    using StatusWord =
        std::conditional_t<sizeof(T) == 8, ulong, std::conditional_t<sizeof(T) == 4, uint, std::conditional_t<sizeof(T) == 2, ushort, uchar>>>;

    using TxnWord =
        std::conditional_t<sizeof(T) == 8, ulong2, std::conditional_t<sizeof(T) == 4, uint2, uint>>;

    struct TileDescriptor
    {
        StatusWord status;
        T          value;
    };

    enum
    {
        TILE_STATUS_PADDING = 32,
    };

    TxnWord* d_tile_descriptions;

    static constexpr size_t description_bytes_per_tile = sizeof(TxnWord);
    static constexpr size_t payload_bytes_per_tile     = 0;

    ScanTileState()
        : d_tile_descriptions{nullptr}
    {
    }

    void SetInclusive(Int tile_index, Var<T> tile_inclusive) {}
    void SetPartial(Int tile_index, Var<T> tile_partial) {}
    void WaitForValid(Int tile_index, Var<StatusWord>& status, Var<T>& value) {}
};

// For multi-word type(key-value pair)
template <typename T>
struct ScanTileState<T, false>
{
};

;
// Decoupled look-back(warp)
template <typename T, typename ScanOpT, typename DelayConstructorT, typename ScanTileStateT = ScanTileState<T>>
class TilePrefixCallbackOp : public LuisaModule
{
  public:
    using WarpReduceT = WarpReduce<T, 32>;

    struct TempStorage
    {
        SmemTypePtr<T> exclusive_prefix;
        SmemTypePtr<T> inclusive_prefix;
        SmemTypePtr<T> block_aggregate;
    };

    using StatusWord = typename ScanTileStateT::StatusWord;


    TempStorage&    temp_storage;
    ScanTileStateT& tile_status;
    ScanOpT         scan_op;
    int             tile_index;
    Var<T>          exclusive_prefix;
    Var<T>          inclusive_prefix;

    TilePrefixCallbackOp(ScanTileStateT& tile_state, TempStorage& temp_storage, ScanOpT scan_op, int tile_index)
        : tile_status{tile_state}
        , temp_storage{temp_storage}
        , scan_op{scan_op}
        , tile_index{tile_index} {};

    TilePrefixCallbackOp(ScanTileStateT& tile_state, TempStorage& temp_storage, ScanOpT scan_op)
        : TilePrefixCallbackOp(
              tile_state, temp_storage, scan_op, compute::block_id().x) {};

  public:
    // core function
    Var<T> operator()(Var<T> block_aggregate)
    {
        $if(thread_x() == 0)
        {
            (*temp_storage.block_aggregate)[0] = block_aggregate;

            tile_status.SetPartial(tile_index, block_aggregate);
        };

        Int             predecessor_idx = tile_index - thread_x() - 1;
        Var<StatusWord> predecessor_status;
        Var<T>          windows_aggregate;

        // decay
        DelayConstructorT construct_delay(tile_index);
        process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay);

        exclusive_prefix = windows_aggregate;

        $while(warp_active_all(predecessor_status != StatusWord(ScanTileStatus::SCAN_TILE_INCLUSIVE)))
        {
            predecessor_idx -= Int(warp_lane_count());
            process_windows(predecessor_idx, predecessor_status, windows_aggregate, construct_delay);
            exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
        };

        $if(thread_x() == 0)
        {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            tile_status.SetInclusive(tile_index, inclusive_prefix);

            (*temp_storage.exclusive_prefix)[0] = exclusive_prefix;
            (*temp_storage.inclusive_prefix)[0] = inclusive_prefix;
        };

        return exclusive_prefix;
    }

  private:
    template <typename DeLayT>
    void process_windows(Int              predecessor_idx,
                         Var<StatusWord>& predecessor_status,
                         Var<T>&          windows_aggregate,
                         DeLayT           delay)
    {
        Var<T> value;
        tile_status.WaitForValid(predecessor_idx, predecessor_status, value);

        Int tial_flag =
            select(0, 1, predecessor_status == StatusWord(ScanTileStatus::SCAN_TILE_INCLUSIVE));

        windows_aggregate =
            WarpReduceT().Reduce(value, scan_op, UInt(warp_lane_count()) * UInt(tial_flag));
    }
};


}  // namespace luisa::parallel_primitive