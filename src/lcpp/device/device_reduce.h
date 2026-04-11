/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 17:31:05
 */

#pragma once
#include <lcpp/common/grid_even_shared.h>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <limits>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/reduce.h>
#include <lcpp/device/details/reduce_by_key.h>
namespace luisa::parallel_primitive
{

using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceReduce : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;

    bool   m_created = false;

  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

#ifndef NDEBUG
    Stream* m_debug_stream; // bind debug stream for sync
#endif
    inline Stream* debug_stream() noexcept
    {
#ifndef NDEBUG
        return m_debug_stream;
#else
        return nullptr;
#endif
    }
    void create(Device& device, Stream* debug_stream = nullptr)
    {
        m_device                   = device;
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for Reduce / Sum / Min / Max / TransformReduce
    template <typename Type4Byte>
    static size_t GetTempStorageBytes(size_t num_item)
    {
        size_t temp_count = 0;
        get_temp_size_scan(temp_count, BLOCK_SIZE, ITEMS_PER_THREAD, num_item);
        return temp_count * sizeof(Type4Byte);
    }

    /// Temp storage bytes for ArgMin / ArgMax
    /// Needs: IndexValuePairT<Type4Byte> temp for reduce + d_in_kv(num_item) + d_out_kv(1)
    template <typename Type4Byte>
    static size_t GetArgTempStorageBytes(size_t num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t reduce_temp_count = 0;
        get_temp_size_scan(reduce_temp_count, BLOCK_SIZE, ITEMS_PER_THREAD, num_item);
        size_t bytes = 0;
        bytes += reduce_temp_count * sizeof(IVP);          // reduce temp
        bytes  = align_up_uint(bytes, alignof(IVP));
        bytes += num_item * sizeof(IVP);                   // d_in_kv
        bytes  = align_up_uint(bytes, alignof(IVP));
        bytes += 1 * sizeof(IVP);                          // d_out_kv
        return bytes;
    }

    /// Temp storage bytes for ReduceByKey
    template <typename KeyType, typename ValueType>
    static size_t GetReduceByKeyTempStorageBytes(size_t num_elements)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * BLOCK_SIZE)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t bytes = 0;
        bytes += tile_count * sizeof(uint);               // tile_states
        bytes  = align_up_uint(bytes, alignof(FlagValuePairT));
        bytes += tile_count * sizeof(FlagValuePairT);     // tile_partial
        bytes  = align_up_uint(bytes, alignof(FlagValuePairT));
        bytes += tile_count * sizeof(FlagValuePairT);     // tile_inclusive
        return bytes;
    }

    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        size_t temp_count = 0;
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item);
        auto temp_view = temp_storage.subview(0, temp_count * sizeof(Type4Byte) / sizeof(uint)).template as<Type4Byte>();

        lcpp_check(reduce_array_recursive<Type4Byte>(
            cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, initial_value, IdentityOp()),
        cmdlist, debug_stream());
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&                           cmdlist,
                BufferView<uint>                       temp_storage,
                BufferView<IndexValuePairT<Type4Byte>> d_in,
                BufferView<IndexValuePairT<Type4Byte>> d_out,
                size_t                                 num_item,
                ReduceOp                               reduce_op,
                IndexValuePairT<Type4Byte>             initial_value)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t temp_count = 0;
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item);
        auto temp_view = temp_storage.subview(0, temp_count * sizeof(IVP) / sizeof(uint)).template as<IVP>();

        lcpp_check(reduce_array_recursive<IVP>(
            cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, initial_value, IdentityOp()),
        cmdlist, debug_stream());
    }


    template <NumericT Type4Byte>
    void Sum(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, SumOp(), Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void Min(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, MinOp(), std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, MaxOp(), std::numeric_limits<Type4Byte>::min());
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t reduce_temp_count = 0;
        get_temp_size_scan(reduce_temp_count, m_block_size, ITEMS_PER_THREAD, num_item);

        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = reduce_temp_count * sizeof(IVP) / sizeof(uint);
        auto reduce_temp = temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(IVP));

        // d_in_kv
        size_t in_kv_uint_count = num_item * sizeof(IVP) / sizeof(uint);
        auto d_in_kv = temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(IVP));

        // d_out_kv
        size_t out_kv_uint_count = 1 * sizeof(IVP) / sizeof(uint);
        auto d_out_kv = temp_storage.subview(offset_bytes / sizeof(uint), out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv), cmdlist, debug_stream());

        Reduce(cmdlist,
               temp_storage.subview(0, reduce_temp_uint_count),
               d_in_kv,
               d_out_kv,
               num_item,
               ArgMinOp(),
               IVP{std::numeric_limits<uint>::max(),
                   std::numeric_limits<Type4Byte>::max()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<Type4Byte>(cmdlist, d_out_kv, d_out, d_index_out), cmdlist, debug_stream());
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t reduce_temp_count = 0;
        get_temp_size_scan(reduce_temp_count, m_block_size, ITEMS_PER_THREAD, num_item);

        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = reduce_temp_count * sizeof(IVP) / sizeof(uint);
        auto reduce_temp = temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(IVP));

        // d_in_kv
        size_t in_kv_uint_count = num_item * sizeof(IVP) / sizeof(uint);
        auto d_in_kv = temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(IVP));

        // d_out_kv
        size_t out_kv_uint_count = 1 * sizeof(IVP) / sizeof(uint);
        auto d_out_kv = temp_storage.subview(offset_bytes / sizeof(uint), out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv), cmdlist, debug_stream());

        Reduce(cmdlist,
               temp_storage.subview(0, reduce_temp_uint_count),
               d_in_kv,
               d_out_kv,
               num_item,
               ArgMaxOp(),
               IVP{0, std::numeric_limits<Type4Byte>::min()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<Type4Byte>(cmdlist, d_out_kv, d_out, d_index_out), cmdlist, debug_stream());
    }


    template <NumericT KeyType, NumericT ValueType, typename ReduceOp>
    void ReduceByKey(CommandList&          cmdlist,
                     BufferView<uint>      temp_storage,
                     BufferView<KeyType>   d_keys_in,
                     BufferView<ValueType> d_values_in,
                     BufferView<KeyType>   d_unique_out,
                     BufferView<ValueType> g_aggregates_out,
                     BufferView<uint>      g_num_runs_out,
                     ReduceOp              reduce_op,
                     size_t                num_elements)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        luisa::vector<luisa::uint> zero_data(1, 0);
        cmdlist << g_num_runs_out.copy_from(zero_data.data());
        int num_tiles = imax(1, (int)ceil((float)num_elements / (ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_states: tile_count * uint
        auto tile_states = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(FlagValuePairT));

        // tile_partial: tile_count * FlagValuePairT
        size_t partial_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(FlagValuePairT));

        // tile_inclusive: tile_count * FlagValuePairT
        size_t inclusive_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_inclusive = temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();

        lcpp_check(
            reduce_by_key_array<KeyType, ValueType>(
                cmdlist,
                tile_states,
                tile_partial,
                tile_inclusive,
                d_keys_in,
                d_values_in,
                d_unique_out,
                g_aggregates_out,
                g_num_runs_out,
                reduce_op,
                num_elements),
             cmdlist, debug_stream());
    }


    template <typename Type4Byte, typename ReduceOp, typename TransformOp>
    void TransformReduce(CommandList&          cmdlist,
                         BufferView<uint>      temp_storage,
                         BufferView<Type4Byte> d_in,
                         BufferView<Type4Byte> d_out,
                         size_t                num_item,
                         ReduceOp              reduce_op,
                         TransformOp           transform_op,
                         Type4Byte             init)
    {
        size_t temp_count = 0;
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item);
        auto temp_view = temp_storage.subview(0, temp_count * sizeof(Type4Byte) / sizeof(uint)).template as<Type4Byte>();
        lcpp_check(
            reduce_array_recursive<Type4Byte, ReduceOp, TransformOp>(
            cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, init, transform_op),
        cmdlist, debug_stream());
    }

  private:
    template <NumericT Type4Byte>
    [[nodiscard]] int arg_construct(CommandList& cmdlist, BufferView<Type4Byte> d_in, BufferView<IndexValuePairT<Type4Byte>> d_kv_out) noexcept
    {
        using ArgReduce          = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgConstructShader = ArgReduce::ArgConstructShaderT;
        auto key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            auto shader = ArgReduce().compile_arg_construct_shader(m_device);
            if(!shader) { return -1; }
            ms_arg_construct_map.try_emplace(key, std::move(shader));
            ms_arg_construct_it = ms_arg_construct_map.find(key);
        }
        if(ms_arg_construct_it == ms_arg_construct_map.end()) { return -1; }
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShader*>(&(*ms_arg_construct_it->second));
        if(!ms_arg_construct_ptr) { return -1; }
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(d_in.size());
        return 0;
    }

    template <NumericT Type4Byte>
    [[nodiscard]] int arg_assign(CommandList&                           cmdlist,
                    BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                    BufferView<Type4Byte>                  d_value_out,
                    BufferView<uint>                       d_index_out) noexcept
    {
        using ArgReduce       = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgAssignShader = ArgReduce::ArgAssignShaderT;
        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_assign_shader(m_device);
            if(!shader) { return -1; }
            ms_arg_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_assign_map.find(key);
        }
        if(ms_arg_assign_it == ms_arg_assign_map.end()) { return -1; }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        if(!ms_arg_assign_ptr) { return -1; }
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_value_out, d_index_out).dispatch(d_index_out.size());
        return 0;
    }

    template <NumericTOrKeyValuePairT Type, typename ReduceOp, typename TransformOp = IdentityOp>
    [[nodiscard]] int reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                BufferView<Type>             temp_storage,
                                BufferView<Type>             arr_in,
                                BufferView<Type>             arr_out,
                                uint                         num_items,
                                uint                         offset,
                                uint                         level,
                                ReduceOp                     reduce_op,
                                Type                         init,
                                TransformOp                  transform_op = IdentityOp()) noexcept
    {
        uint tile_items = m_block_size * ITEMS_PER_THREAD;
        uint num_tiles  = imax(1, (uint)ceil((float)num_items / tile_items));

        // reduce_device_occupancy × subscription_factor
        // reduce_device_occupancy = sm_occupancy × sm_count
        // sm_occupancy = 8 sm_count = 128, subscription_factor = 5
        constexpr auto max_blocks = BLOCK_SIZE * 8 * 5;

        using ReduceShader           = details::ReduceModule<Type, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceKernel           = ReduceShader::ReduceShaderKernel;
        using ReduceSingleTileShader = ReduceShader::ReduceSingleTileShaderKernel;

        size_t           size_elements     = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);

        if(num_tiles > 1)
        {
            auto key          = get_type_and_op_desc<Type>(reduce_op, transform_op);
            auto ms_reduce_it = ms_reduce_map.find(key);
            if(ms_reduce_it == ms_reduce_map.end())
            {
                auto shader = ReduceShader().compile(m_device, m_shared_mem_size, reduce_op, transform_op);
                ms_reduce_map.try_emplace(key, std::move(shader));
                ms_reduce_it = ms_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_reduce_it->second));

            GridEvenShared even_share;
            even_share.DispatchInit(num_items, max_blocks, tile_items);
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_items, even_share).dispatch(m_block_size * num_tiles);
            lcpp_check(
                reduce_array_recursive<Type>(
                    cmdlist, temp_buffer_level, temp_buffer_level, arr_out, num_tiles, num_tiles, level + 1, reduce_op, init, transform_op),
                cmdlist, debug_stream());
        }
        else
        {
            auto key          = get_type_and_op_desc<Type>(reduce_op, IdentityOp());
            auto ms_reduce_it = ms_single_reduce_map.find(key);
            if(ms_reduce_it == ms_single_reduce_map.end())
            {
                auto shader =
                    ReduceShader().compile_single_tile(m_device, m_shared_mem_size, reduce_op, IdentityOp());
                ms_single_reduce_map.try_emplace(key, std::move(shader));
                ms_reduce_it = ms_single_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceSingleTileShader*>(&(*ms_reduce_it->second));
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_items, init).dispatch(m_block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
        return 0;
    };


    template <NumericT KeyType, NumericT ValueType, typename ReduceOp, typename FlagValuePairT = KeyValuePair<int, ValueType>>
    [[nodiscard]] int reduce_by_key_array(luisa::compute::CommandList& cmdlist,
                             BufferView<uint>             tile_states,
                             BufferView<FlagValuePairT>   tile_partial,
                             BufferView<FlagValuePairT>   tile_inclusive,
                             BufferView<KeyType>          keys_in,
                             BufferView<ValueType>        values_in,
                             BufferView<KeyType>          unique_out,
                             BufferView<ValueType>        aggregated_out,
                             BufferView<uint>             num_runs_out,
                             ReduceOp                     reduce_op,
                             uint                         num_items) noexcept
    {
        uint tile_items = m_block_size * ITEMS_PER_THREAD;
        uint num_tiles  = imax(1, (uint)ceil((float)num_items / tile_items));


        using ReduceByKey = details::ReduceByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceByKeyTileStateInitKernel = ReduceByKey::ScanTileStateInitKernel;
        using ReduceByKeyKernel              = ReduceByKey::ReduceByKeyKernel;

        // init
        auto init_key                   = get_type_and_op_desc<KeyType, ValueType>();
        auto ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        if(ms_scan_tile_state_init_it == ms_scan_tile_state_init_map.end())
        {
            auto shader = ReduceByKey().compile_scan_tile_state_init(m_device);
            if (!shader) {
                return -1; // shader create failed
            }
            ms_scan_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ReduceByKeyTileStateInitKernel*>(&(*ms_scan_tile_state_init_it->second));
        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, num_tiles)
                       .dispatch(num_tiles * m_block_size);
        // reduce by key
        auto key                 = get_type_and_op_desc<KeyType, ValueType>(reduce_op);
        auto ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        if(ms_reduce_by_key_it == ms_reduce_by_key_map.end())
        {
            auto shader = ReduceByKey().compile(m_device, m_shared_mem_size, reduce_op);
            ms_reduce_by_key_map.try_emplace(key, std::move(shader));
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr = reinterpret_cast<ReduceByKeyKernel*>(&(*ms_reduce_by_key_it->second));

        cmdlist << (*ms_reduce_by_key_ptr)(
                       tile_states, tile_partial, tile_inclusive, keys_in, values_in, unique_out, aggregated_out, num_runs_out, num_items)
                       .dispatch(m_block_size * num_tiles);
        return 0;
    };


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_single_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_transform_reduce_map;
    // for arg reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    // for reduce by key
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_tile_state_init_map;
};
}  // namespace luisa::parallel_primitive
