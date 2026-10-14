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
#include <lcpp/runtime/interop_dispatch.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/shader_cache.h>
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

// DeviceReduce policy default (epoch-4): BLOCK_SIZE x ITEMS_PER_THREAD tile.
// Tuned on RTX 5060 (4M int32 Sum) after coalescing ConsumeFullTile loads to a
// block-striped layout: 256x8 is the best *portable* policy (cuda 0.022 ms
// ~1.11x CUB, dx 0.039 ms ~0.87x, vk 0.036 ms ~0.77x) and, unlike 256x16, keeps
// the shared-memory footprint under the 32 KiB DX/HLSL floor for the default
// *wide* element type used by ArgMin/ArgMax (IndexValuePairT, 8 bytes):
//   (256*8 + 256*8/32) * 8 = 2112 * 8 = 16896 B <= 32 KiB
//   (256*16 + 256*16/32)* 8 = 4224 * 8 = 33792 B >  32 KiB  (would break dx)
  // So ITEMS=16 matches cuda parity but is rejected for the all-type default.
  static constexpr size_t REDUCE_ITEMS_PER_THREAD = 8;
// Epoch D -- interop-only tile override.
//
// The 32 KiB HLSL floor above binds the *DX/vk SPIR-V* codegen, not the
// cuda-compiled kernels the vk-cuda interop path imports.  For the plain
// narrow-element reductions the interop path can therefore use the wider
// 256x16 tile that epoch 4 rejected as a portable default.  Measured on
// RTX 5060 / vk (4,194,304 int32 Sum, LCPP_INTEROP=reduce, canonical
// lcpp_cub_bench, batch=100, 3 rounds): 256x16 ties 256x8 at 4M and is 1.45 %
// faster at 16M (0.1731 vs 0.1757 ms, non-overlapping round medians); the
// cuda-backend kernel itself is 0.8 % (bench) / 2.6 % (pipelined harness)
// faster at 16M; at 4M only the pipelined harness resolves it (10-19 %), the
// canonical bench shows a tie there (see .kimix_cache/epoch_D_report.md).
//
// The gate is deliberately narrow (see reduce_interop_wide_tile_v):
//   * interop must be active,
//   * the element type must be an integral type of at most 4 bytes, and
//   * the reduction must have an identity *transform*.
// ArgMin/ArgMax reduce 8-byte IndexValuePairT elements whose *tie-breaking*
// depends on the association order, any floating-point reduction is not
// associative, and a transforming reduction changes recursion structure when
// the tile changes (the single-tile terminal kernel is compiled with the
// identity transform -- pre-existing lcpp behaviour, documented by
// vk_cuda_interop_reduce_test), so all of those keep the portable tile, their
// current results, and the same recursion structure on every path.  This is
// also what keeps every DX/HLSL kernel at the tile it has today, since only
// the interop branch below ever instantiates the wider one.
//
// No call site changes: GetTempStorageBytes<Type4Byte>() keeps sizing the
// caller's temp storage with the portable tile, which is an upper bound of the
// wider tile's requirement (a wider tile produces fewer per-level blocks, and
// the per-level regions are exactly num_blocks elements).
  static constexpr size_t REDUCE_INTEROP_ITEMS_PER_THREAD = 16;
  /// True when `Type` may use the wider interop tile (see above).
  template <typename Type>
static constexpr bool reduce_interop_wide_tile_v =
    std::is_integral_v<Type> && sizeof(Type) <= 4u;

// ReduceByKey runs a dedicated tile, independent of REDUCE_ITEMS_PER_THREAD.
// Its kernel allocates THREE shared arrays each sized by the tile element count
// (s_keys, s_values, and the two-phase-scatter s_scatter_keys at 8 B), so its
// DX/HLSL 32 KiB floor binds ~3x sooner than the single-array Sum/Reduce path.
// 256x8 would need (256*8 + 256*8/32) * (4 + 4 + 8) = 33792 B > 32 KiB and fail
// on dx.  Freeze reduce-by-key at the epoch-3 validated 256x4 tile so its
// behavior, results and shared footprint are unchanged from the committed
// baseline, while the hot reduce ops use the tuned REDUCE_ITEMS_PER_THREAD.
static constexpr size_t REDUCE_BY_KEY_ITEMS_PER_THREAD = 4;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = REDUCE_ITEMS_PER_THREAD>
class DeviceReduce : public LuisaModule
{
    private:
      uint m_block_size = BLOCK_SIZE;
      uint m_warp_nums  = WARP_NUMS;
      uint m_shared_mem_size = 0;
      // Epoch D: shared footprint of the wider, interop-only tile.
      uint m_interop_shared_mem_size = 0;
      uint m_rbk_shared_mem_size = 0;
      Device m_device;
      // Optional vk-cuda interop sink (epoch A).  Null => pure-vulkan DSL path.
      InteropContext* m_interop = nullptr;

      bool m_created = false;

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
    // `interop` is optional: pass a valid InteropContext (see
    // lcpp/runtime/interop_dispatch.h) to launch the plain Sum/Min/Max/
    // TransformReduce kernels as cuda-backend modules inside the vk command
    // buffer.  Without it (or when the context is null because the fast path
    // is unavailable) every op keeps its pure-vulkan DSL dispatch path.
    void create(Device& device, Stream* debug_stream = nullptr, InteropContext* interop = nullptr)
    {
        m_device  = device;
        // Epoch C per-op policy: `LCPP_INTEROP` (1|all|auto|a,b,c) decides
        // per module whether this one takes the fast path; see
        // lcpp/runtime/interop_dispatch.h.  `auto` keeps this module on the
        // measured-faster path.
        m_interop = interop_for(interop, "reduce");
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space = num_elements_per_block / m_warp_nums;
        m_shared_mem_size = (num_elements_per_block + extra_space);
        // Epoch D: the wider tile the interop fast path may use for narrow
        // integral elements (see REDUCE_INTEROP_ITEMS_PER_THREAD).
        int interop_elements_per_block = m_block_size * REDUCE_INTEROP_ITEMS_PER_THREAD;
        m_interop_shared_mem_size =
            interop_elements_per_block + interop_elements_per_block / m_warp_nums;
        // reduce-by-key uses its own (smaller, DX-floor-bound) tile
        int rbk_elements_per_block = m_block_size * REDUCE_BY_KEY_ITEMS_PER_THREAD;
        m_rbk_shared_mem_size      = rbk_elements_per_block + rbk_elements_per_block / m_warp_nums;
        m_created                  = true;
    }

    /// True when this module dispatches its plain reduce kernels through the
    /// vk-cuda interop fast path (cuda-compiled module, vk command buffer).
    [[nodiscard]] bool interop_active() const noexcept { return m_interop != nullptr; }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for Reduce / Sum / Min / Max / TransformReduce
    template <typename Type4Byte>
    static size_t GetTempStorageBytes(size_t num_item)
    {
        size_t temp_count = 0;
        get_temp_size_scan(temp_count, BLOCK_SIZE, ITEMS_PER_THREAD, num_item, sizeof(Type4Byte));
        return temp_count * sizeof(Type4Byte);
    }

    /// Temp storage bytes for ArgMin / ArgMax
    /// Needs: IndexValuePairT<Type4Byte> temp for reduce + d_in_kv(num_item) + d_out_kv(1)
    template <typename Type4Byte>
    static size_t GetArgTempStorageBytes(size_t num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        size_t reduce_temp_count = 0;
        get_temp_size_scan(reduce_temp_count, BLOCK_SIZE, ITEMS_PER_THREAD, num_item, sizeof(IVP));
        size_t bytes = 0;
        bytes += reduce_temp_count * sizeof(IVP);          // reduce temp
        bytes  = align_up_view(bytes);
        bytes += num_item * sizeof(IVP);                   // d_in_kv
        bytes  = align_up_view(bytes);
        bytes += 1 * sizeof(IVP);                          // d_out_kv
        return bytes;
    }

    /// Temp storage bytes for ReduceByKey
    template <typename KeyType, typename ValueType>
    static size_t GetReduceByKeyTempStorageBytes(size_t num_elements)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        int num_tiles = imax(1, (int)ceil((float)num_elements / (REDUCE_BY_KEY_ITEMS_PER_THREAD * BLOCK_SIZE)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t bytes = 0;
        bytes += tile_count * sizeof(uint);               // tile_states
        bytes  = align_up_view(bytes);
        bytes += tile_count * sizeof(FlagValuePairT);     // tile_partial
        bytes  = align_up_view(bytes);
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
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item, sizeof(Type4Byte));
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
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item, sizeof(IVP));
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
        get_temp_size_scan(reduce_temp_count, m_block_size, ITEMS_PER_THREAD, num_item, sizeof(IVP));

        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = reduce_temp_count * sizeof(IVP) / sizeof(uint);
        auto reduce_temp = temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // d_in_kv
        size_t in_kv_uint_count = num_item * sizeof(IVP) / sizeof(uint);
        auto d_in_kv = temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

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
        get_temp_size_scan(reduce_temp_count, m_block_size, ITEMS_PER_THREAD, num_item, sizeof(IVP));

        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = reduce_temp_count * sizeof(IVP) / sizeof(uint);
        auto reduce_temp = temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // d_in_kv
        size_t in_kv_uint_count = num_item * sizeof(IVP) / sizeof(uint);
        auto d_in_kv = temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

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
        auto ptr = luisa::allocate_with_allocator<luisa::uint>(0); // retain until the cmdlist commit
        cmdlist << g_num_runs_out.copy_from(luisa::span<luisa::uint>({ptr, 1}));
        cmdlist.add_dtor_callback([ptr] {
            luisa::deallocate_with_allocator(ptr);
        });
        
        int num_tiles = imax(1, (int)ceil((float)num_elements / (REDUCE_BY_KEY_ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_states: tile_count * uint
        auto tile_states = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_partial: tile_count * FlagValuePairT
        size_t partial_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

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
        get_temp_size_scan(temp_count, m_block_size, ITEMS_PER_THREAD, num_item, sizeof(Type4Byte));
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
        auto key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>());
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
        auto launched = false;
        // ---- interop fast path: index/value pair construction ----
        // Usage table: d_in READ (one element per dispatch thread), d_kv_out
        // WRITE (one IndexValuePairT per element).
        if(m_interop != nullptr)
        {
            auto it = ms_arg_construct_interop_map.find(key);
            if(it == ms_arg_construct_interop_map.end())
            {
                it = import_into_cache(
                    ms_arg_construct_interop_map,
                    key,
                    ArgReduce().compile_arg_construct_shader(m_interop->cuda_device()));
            }
            if(it != ms_arg_construct_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<IndexValuePairT<Type4Byte>>, Usage::WRITE>>();
                cmdlist << kernel(d_in, d_kv_out).dispatch(static_cast<uint>(d_in.size()));
                launched = true;
            }
        }
        if(!launched)
        {
            cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(d_in.size());
        }
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
        auto key              = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>());
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
        auto launched = false;
        // ---- interop fast path: split the (index, value) pair ----
        // Usage table: d_kv_in READ, d_value_out WRITE, d_index_out WRITE.
        if(m_interop != nullptr)
        {
            auto it = ms_arg_assign_interop_map.find(key);
            if(it == ms_arg_assign_interop_map.end())
            {
                it = import_into_cache(
                    ms_arg_assign_interop_map,
                    key,
                    ArgReduce().compile_arg_assign_shader(m_interop->cuda_device()));
            }
            if(it != ms_arg_assign_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<IndexValuePairT<Type4Byte>>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::WRITE>>();
                cmdlist << kernel(d_kv_in, d_value_out, d_index_out).dispatch(static_cast<uint>(d_index_out.size()));
                launched = true;
            }
        }
        if(!launched)
        {
            cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_value_out, d_index_out).dispatch(d_index_out.size());
        }
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
        // Align the region base: the recursion passes subviews of the temp
        // buffer as kernel arguments, whose byte offsets must satisfy the
        // descriptor-view alignment (legacy Vulkan HLSL path).
        static_assert(descriptor_view_alignment % sizeof(Type) == 0,
                      "temp carving requires the element size to divide the descriptor alignment");
        offset = static_cast<uint>(align_up_view(static_cast<size_t>(offset) * sizeof(Type)) / sizeof(Type));

        // Epoch D: the interop fast path may run a wider tile than the portable
        // one (see REDUCE_INTEROP_ITEMS_PER_THREAD).  `kInteropItems` is a
        // compile-time constant, so the module type below is either the wider
        // instantiation (narrow integral elements, i.e. the measured case) or --
        // for every wider / order-sensitive element type -- simply the same one
        // the DSL path uses, with no extra instantiation.
        //
        // Only a plain (identity-transform) reduction may change tile: the
        // single-tile terminal kernel is compiled with IdentityOp, so a
        // non-identity transform is only applied while the recursion is
        // multi-tile (pre-existing lcpp behaviour, documented by
        // vk_cuda_interop_reduce_test).  Transforming reductions therefore keep
        // the portable tile so both dispatch paths keep the same recursion
        // structure -- and the same result.
        constexpr bool   kInteropWide  = reduce_interop_wide_tile_v<Type> &&
                                         std::is_same_v<TransformOp, IdentityOp>;
        constexpr size_t kInteropItems = kInteropWide ? REDUCE_INTEROP_ITEMS_PER_THREAD
                                        : ITEMS_PER_THREAD;
        // reduce_device_occupancy × subscription_factor
        // reduce_device_occupancy = sm_occupancy × sm_count
        // sm_occupancy = 8 sm_count = 128, subscription_factor = 5
        constexpr auto max_blocks = BLOCK_SIZE * 8 * 5;
        using ReduceShader           = details::ReduceModule<Type, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceKernel           = ReduceShader::ReduceShaderKernel;
        using ReduceSingleTileShader = ReduceShader::ReduceSingleTileShaderKernel;
        // The shader *type* does not depend on the tile (the kernel signature is
        // unchanged), so the interop branch can use its own instantiation and
        // still hand the result to the same import cache.
        using InteropReduceShader = details::ReduceModule<Type, BLOCK_SIZE, kInteropItems>;
        // Epoch B: every element type whose *values* the interop ABI can carry
        // by value in one 16-byte-aligned `Params` slot takes the fast path.
        // That includes the DSL struct elements used by ArgMin/ArgMax
        // (IndexValuePairT) and ReduceByKey (KeyValuePair): their host-side
        // layout is a plain aggregate of scalars whose size/alignment the cuda
        // backend's `struct alignas(16) Params` member mirrors exactly (see the
        // ABI assertions in tests/vk_cuda_interop_ops_test.cpp).
        constexpr bool interop_eligible = std::is_trivially_copyable_v<Type>;

        size_t size_elements = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);

        // Epoch D (review fix): a call must never *mix* geometries.  The import
        // below can fail closed on this very call (risk R3): it then clears
        // m_interop and the launch does not happen, so the portable tile has to
        // take over completely -- its own num_tiles, its own dispatch geometry
        // and the size it hands to the recursion.  The two geometries are
        // therefore computed once each, the interop one is only *committed* when
        // its launch really happened, and the fallback re-derives the portable
        // one.  `tile_items` / `num_tiles` are the committed geometry for the
        // rest of this call.
        const uint interop_tile_items = m_block_size * static_cast<uint>(kInteropItems);
        const uint dsl_tile_items     = m_block_size * static_cast<uint>(ITEMS_PER_THREAD);
        auto       launched           = false;
        uint       tile_items         = dsl_tile_items;  // portable: the safe default
        uint       num_tiles          = imax(1, (uint)ceil((float)num_items / tile_items));

        if constexpr(interop_eligible)
        {
            if(m_interop != nullptr)
            {
                const uint wide_tiles = imax(1, (uint)ceil((float)num_items / interop_tile_items));
                if(wide_tiles > 1)
                {
                    auto key = get_type_and_op_desc<Type>(reduce_op, transform_op);
                    auto it  = ms_reduce_interop_map.find(key);
                    if(it == ms_reduce_interop_map.end())
                    {
                        it = import_into_interop_cache(
                            m_interop,
                            ms_reduce_interop_map,
                            key,
                            InteropReduceShader().compile(
                                m_interop->cuda_device(), m_interop_shared_mem_size, reduce_op, transform_op));
                    }
                    if(it != ms_reduce_interop_map.end())
                    {
                        GridEvenShared even_share;
                        even_share.DispatchInit(num_items, max_blocks, interop_tile_items);
                        // Usages are baked into the signature: the thread-level
                        // kernel only reads d_in, and writes the per-block
                        // aggregate into d_out (which aliases d_in on the
                        // recursive levels, hence READ_WRITE).
                        auto kernel = it->second.shader.template kernel<
                            vk_cuda_interop::CudaArg<Buffer<Type>, Usage::READ>,
                            vk_cuda_interop::CudaArg<Buffer<Type>, Usage::READ_WRITE>,
                            uint,
                            GridEvenShared>();
                        cmdlist << kernel(arr_in, temp_buffer_level, num_items, even_share)
                                       .dispatch(m_block_size * wide_tiles);
                        launched   = true;
                        tile_items = interop_tile_items;
                        num_tiles  = wide_tiles;
                    }
                }
                else
                {
                    auto key = get_type_and_op_desc<Type>(reduce_op, IdentityOp());
                    auto it  = ms_single_reduce_interop_map.find(key);
                    if(it == ms_single_reduce_interop_map.end())
                    {
                        it = import_into_interop_cache(
                            m_interop,
                            ms_single_reduce_interop_map,
                            key,
                            InteropReduceShader().compile_single_tile(
                                m_interop->cuda_device(), m_interop_shared_mem_size, reduce_op, IdentityOp()));
                    }
                    if(it != ms_single_reduce_interop_map.end())
                    {
                        auto kernel = it->second.shader.template kernel<
                            vk_cuda_interop::CudaArg<Buffer<Type>, Usage::READ>,
                            vk_cuda_interop::CudaArg<Buffer<Type>, Usage::READ_WRITE>,
                            uint,
                            Type>();
                        // non-recursive
                        cmdlist << kernel(arr_in, temp_buffer_level, num_items, init).dispatch(m_block_size);
                        // Epoch D: `num_tiles == 1` guarantees the reduction
                        // result is temp_buffer_level[0]; copying exactly one
                        // element keeps this correct for any temp storage that
                        // is *larger* than the layout this recursion needs
                        // (which is what an interop-only wider tile produces --
                        // its per-level regions are smaller).
                        cmdlist << arr_out.copy_from(temp_buffer_level.subview(0, 1));
                        launched   = true;
                        tile_items = interop_tile_items;
                        num_tiles  = wide_tiles;
                    }
                }
            }
        }

        if(!launched)
        {
            // Portable tile -- also the fail-closed fallback, in which case any
            // geometry committed above is discarded (nothing was recorded) and
            // re-derived here, so the narrow shader is never launched with wide
            // geometry and the recursion never inherits a mixed size.
            tile_items = dsl_tile_items;
            num_tiles  = imax(1, (uint)ceil((float)num_items / tile_items));
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
                cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_items, even_share)
                               .dispatch(m_block_size * num_tiles);
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
                // Epoch D: `num_tiles == 1` guarantees the reduction result is
                // temp_buffer_level[0]; copying exactly one element keeps this
                // correct for any temp storage that is *larger* than the layout
                // this recursion needs.
                cmdlist << arr_out.copy_from(temp_buffer_level.subview(0, 1));
            }
        }

        if(num_tiles > 1)
        {
            lcpp_check(
                reduce_array_recursive<Type>(
                    cmdlist, temp_buffer_level, temp_buffer_level, arr_out, num_tiles, num_tiles, level + 1, reduce_op, init, transform_op),
                cmdlist, debug_stream());
        }
        return 0;
    };

    /// Cache-miss path of the interop fast path: import the freshly compiled
    /// cuda-backend shader into the vk device and store it under `key`.
    /// Import fails closed (OptiX/printing/bound-argument kernels); in that
    /// case interop is disabled for this module and the caller falls back to
    /// the DSL path.  Thin alias for `import_into_interop_cache` so the
    /// per-module call sites read the same way they did in Epoch A.
    template <typename ShaderT>
    [[nodiscard]] detail::InteropShaderMap::iterator import_into_cache(detail::InteropShaderMap&         map,
                                                                      const detail::ShaderCacheKey&     key,
                                                                      const luisa::unique_ptr<ShaderT>& compiled_shader) noexcept
    {
        return import_into_interop_cache(m_interop, map, key, compiled_shader);
    }


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
        uint tile_items = m_block_size * REDUCE_BY_KEY_ITEMS_PER_THREAD;
        uint num_tiles  = imax(1, (uint)ceil((float)num_items / tile_items));


        using ReduceByKey = details::ReduceByKeyModule<KeyType, ValueType, BLOCK_SIZE, REDUCE_BY_KEY_ITEMS_PER_THREAD>;
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
        uint init_threads = static_cast<uint>(num_tiles) * m_block_size;
        auto init_launched = false;
        // ---- interop fast path: tile-state init ----
        // Usage table: `InitializeWardStatus` only writes `tile_state` (head
        // padding + per-tile INVALID); READ_WRITE is the conservative
        // declaration (identical ordering to WRITE for a write-only kernel:
        // emplace_write_layer records the read layer too).
        if(m_interop != nullptr)
        {
            auto it = ms_scan_tile_state_init_interop_map.find(init_key);
            if(it == ms_scan_tile_state_init_interop_map.end())
            {
                it = import_into_cache(
                    ms_scan_tile_state_init_interop_map,
                    init_key,
                    ReduceByKey().compile_scan_tile_state_init(m_interop->cuda_device()));
            }
            if(it != ms_scan_tile_state_init_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    uint>();
                cmdlist << kernel(tile_states, uint(num_tiles)).dispatch(init_threads);
                init_launched = true;
            }
        }
        if(!init_launched)
        {
            cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, num_tiles).dispatch(init_threads);
        }
        // reduce by key
        auto key                 = get_type_and_op_desc<KeyType, ValueType>(reduce_op);
        auto ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        if(ms_reduce_by_key_it == ms_reduce_by_key_map.end())
        {
            auto shader = ReduceByKey().compile(m_device, m_rbk_shared_mem_size, reduce_op);
            ms_reduce_by_key_map.try_emplace(key, std::move(shader));
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr = reinterpret_cast<ReduceByKeyKernel*>(&(*ms_reduce_by_key_it->second));

        uint threads   = m_block_size * num_tiles;
        auto launched  = false;
        // ---- interop fast path: decoupled-look-back reduce-by-key ----
        // Usage table: the tile-state triple is the decoupled-look-back channel
        // (volatile_write + WaitForValid spin) -> READ_WRITE; the input
        // key/value arrays are read-only; the unique-key / aggregate outputs
        // and the run counter are written.  `g_num_runs_out` is also written
        // by the ordinary vk H2D copy command that precedes this dispatch in
        // the same mixed command list (mix of interop launches and vk copies is
        // supported and ordered by the reorder pass).
        if(m_interop != nullptr)
        {
            auto it = ms_reduce_by_key_interop_map.find(key);
            if(it == ms_reduce_by_key_interop_map.end())
            {
                it = import_into_cache(
                    ms_reduce_by_key_interop_map,
                    key,
                    ReduceByKey().compile(m_interop->cuda_device(), m_rbk_shared_mem_size, reduce_op));
            }
            if(it != ms_reduce_by_key_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValuePairT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValuePairT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<KeyType>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<KeyType>, Usage::WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::WRITE>,
                    uint>();
                cmdlist << kernel(tile_states,
                                  tile_partial,
                                  tile_inclusive,
                                  keys_in,
                                  values_in,
                                  unique_out,
                                  aggregated_out,
                                  num_runs_out,
                                  uint(num_items))
                               .dispatch(threads);
                launched = true;
            }
        }
        if(!launched)
        {
            cmdlist << (*ms_reduce_by_key_ptr)(
                           tile_states, tile_partial, tile_inclusive, keys_in, values_in, unique_out, aggregated_out, num_runs_out, num_items)
                           .dispatch(threads);
        }
        return 0;
    };


    detail::ShaderCacheMap ms_reduce_map;
    detail::ShaderCacheMap ms_single_reduce_map;
    detail::ShaderCacheMap ms_transform_reduce_map;
    // for arg reduce
    detail::ShaderCacheMap ms_arg_construct_map;
    detail::ShaderCacheMap ms_arg_assign_map;
    // for reduce by key
    detail::ShaderCacheMap ms_reduce_by_key_map;
    detail::ShaderCacheMap ms_scan_tile_state_init_map;
    // epoch A/B: vk-cuda interop kernels shadowing the maps above.  Destroyed
    // with the module => must outlive nothing but m_interop's ext (risk R6).
    detail::InteropShaderMap ms_reduce_interop_map;
    detail::InteropShaderMap ms_single_reduce_interop_map;
    detail::InteropShaderMap ms_arg_construct_interop_map;
    detail::InteropShaderMap ms_arg_assign_interop_map;
    detail::InteropShaderMap ms_reduce_by_key_interop_map;
    detail::InteropShaderMap ms_scan_tile_state_init_interop_map;
};
}  // namespace luisa::parallel_primitive
