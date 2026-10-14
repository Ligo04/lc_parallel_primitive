/*
 * @Author: Ligo 
 * @Date: 2025-10-09 09:52:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:12:25
 */


#pragma once
#include <cstddef>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lcpp/runtime/core.h>
#include <lcpp/runtime/interop_dispatch.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/device/details/single_pass_scan_operator.h>
#include <lcpp/device/details/scan_by_key.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
// Tuned policy for 4-byte element types (epoch 2). The scan kernel uses a
// coalesced block-striped load/store transposed through shared memory (see
// device/details/scan.h). 256 threads x 15 items/tile balances DRAM
// coalescing against decoupled-look-back depth and stays within the DX 32 KiB
// shared-memory floor: (256*15 + 256*15/32)*4 = 15840 B per block.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = 15>
class DeviceScan : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    // Optional vk-cuda interop sink (epoch B).  Null => pure-vulkan DSL path.
    InteropContext* m_interop = nullptr;
    bool   m_created = false;

  public:
    DeviceScan()  = default;
    ~DeviceScan() = default;

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
    // lcpp/runtime/interop_dispatch.h) to launch the scan / scan-by-key
    // kernels as cuda-backend modules inside the vk command buffer.  Without
    // it (or when the context is null because the fast path is unavailable)
    // every op keeps its pure-vulkan DSL dispatch path.
    void create(Device& device, Stream* debug_stream = nullptr, InteropContext* interop = nullptr)
    {
        m_device                   = device;
        // Epoch C per-op policy: `LCPP_INTEROP` (1|all|auto|a,b,c) decides
        // per module whether this one takes the fast path; see
        // lcpp/runtime/interop_dispatch.h.
        m_interop = interop_for(interop, "scan");
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    /// True when this module dispatches its scan kernels through the vk-cuda
    /// interop fast path (cuda-compiled module, vk command buffer).
    [[nodiscard]] bool interop_active() const noexcept { return m_interop != nullptr; }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for ExclusiveScan / InclusiveScan / ExclusiveSum / InclusiveSum
    template <typename Type4Byte>
    static size_t GetTempStorageBytes(size_t num_items)
    {
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * BLOCK_SIZE)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t bytes = 0;
        bytes += tile_count * sizeof(uint);               // tile_status
        bytes  = align_up_view(bytes);
        bytes += tile_count * sizeof(Type4Byte);          // tile_partial
        bytes  = align_up_view(bytes);
        bytes += tile_count * sizeof(Type4Byte);          // tile_inclusive
        return bytes;
    }

    /// Temp storage bytes for ExclusiveScanByKey / InclusiveScanByKey
    template <typename KeyType, typename ValueType>
    static size_t GetScanByKeyTempStorageBytes(size_t num_items)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * BLOCK_SIZE)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t bytes = 0;
        bytes += tile_count * sizeof(uint);                  // tile_status
        bytes  = align_up_view(bytes);
        bytes += tile_count * sizeof(FlagValuePairT);        // tile_partial
        bytes  = align_up_view(bytes);
        bytes += tile_count * sizeof(FlagValuePairT);        // tile_inclusive
        bytes  = align_up_view(bytes);
        bytes += num_tiles * sizeof(KeyType);                // d_prev_keys_in
        return bytes;
    }

    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT Type4Byte, typename ScanOp>
    void ExclusiveScan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        int  num_tiles   = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_status
        auto tile_status = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_partial
        size_t partial_uint_count = tile_count * sizeof(Type4Byte) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<Type4Byte>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_inclusive
        size_t inclusive_uint_count = tile_count * sizeof(Type4Byte) / sizeof(uint);
        auto tile_inclusive = temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<Type4Byte>();

        lcpp_check(scan_array<Type4Byte>(
            cmdlist, tile_status, tile_partial, tile_inclusive, d_in, d_out, num_items, scan_op, initial_value, false),
            cmdlist, debug_stream());
    }

    template <NumericT Type4Byte, typename ScanOp>
    void InclusiveScan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        int  num_tiles   = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_status
        auto tile_status = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_partial
        size_t partial_uint_count = tile_count * sizeof(Type4Byte) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<Type4Byte>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_inclusive
        size_t inclusive_uint_count = tile_count * sizeof(Type4Byte) / sizeof(uint);
        auto tile_inclusive = temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<Type4Byte>();

        lcpp_check(scan_array<Type4Byte>(
            cmdlist, tile_status, tile_partial, tile_inclusive, d_in, d_out, num_items, scan_op, initial_value, true),
            cmdlist, debug_stream());
    }

    template <NumericT Type4Byte>
    void ExclusiveSum(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_items)
    {
        ExclusiveScan(
            cmdlist,
            temp_storage,
            d_in,
            d_out,
            num_items,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void InclusiveSum(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_items)
    {
        InclusiveScan(
            cmdlist,
            temp_storage,
            d_in,
            d_out,
            num_items,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void ExclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_status
        auto tile_status = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_partial
        size_t partial_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_inclusive
        size_t inclusive_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_inclusive = temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();
        offset_bytes += inclusive_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // d_prev_keys_in
        size_t prev_keys_uint_count = bytes_to_uint_count(num_tiles * sizeof(KeyType));
        auto d_prev_keys_in = temp_storage.subview(offset_bytes / sizeof(uint), prev_keys_uint_count).template as<KeyType>();

        lcpp_check(scan_by_key_array<KeyType, ValueType>(cmdlist,
                                              tile_status,
                                              tile_partial,
                                              tile_inclusive,
                                              d_keys_in,
                                              d_prev_keys_in,
                                              d_values_in,
                                              d_values_out,
                                              num_items,
                                              scan_op,
                                              initial_value,
                                              false),
        cmdlist, debug_stream());
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void InclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_status
        auto tile_status = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_partial
        size_t partial_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_partial = temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // tile_inclusive
        size_t inclusive_uint_count = tile_count * sizeof(FlagValuePairT) / sizeof(uint);
        auto tile_inclusive = temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();
        offset_bytes += inclusive_uint_count * sizeof(uint);
        offset_bytes = align_up_view(offset_bytes);

        // d_prev_keys_in
        size_t prev_keys_uint_count = num_tiles * sizeof(KeyType) / sizeof(uint);
        auto d_prev_keys_in = temp_storage.subview(offset_bytes / sizeof(uint), prev_keys_uint_count).template as<KeyType>();

        lcpp_check(scan_by_key_array<KeyType, ValueType>(cmdlist,
                                              tile_status,
                                              tile_partial,
                                              tile_inclusive,
                                              d_keys_in,
                                              d_prev_keys_in,
                                              d_values_in,
                                              d_values_out,
                                              num_items,
                                              scan_op,
                                              initial_value,
                                              true),
        cmdlist, debug_stream());
    }


    template <NumericT Type4Byte>
    void ExclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<Type4Byte> d_keys_in,
                           BufferView<Type4Byte> d_values_in,
                           BufferView<Type4Byte> d_values_out,
                           size_t                num_items)
    {
        ExclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            num_items,
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void InclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<Type4Byte> d_keys_in,
                           BufferView<Type4Byte> d_values_in,
                           BufferView<Type4Byte> d_values_out,
                           size_t                num_items)
    {
        InclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            num_items,
            Type4Byte(0));
    }


  private:
    template <NumericT Type4Byte, typename ScanOp>
    [[nodiscard]] int scan_array(CommandList&          cmdlist,
                    BufferView<uint>      tile_states,
                    BufferView<Type4Byte> tile_partial,
                    BufferView<Type4Byte> tile_inclusive,
                    BufferView<Type4Byte> d_in,
                    BufferView<Type4Byte> d_out,
                    size_t                num_items,
                    ScanOp                scan_op,
                    Type4Byte             initial_value,
                    bool                  is_inclusive)
    {
        uint num_tiles = imax(1, ceil_div(num_items, (ITEMS_PER_THREAD * m_block_size)));

        using ScanShader = details::ScanModule<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanTileStateInitKernel = ScanShader::ScanTileStateInitKernel;
        using ScanShaderKernel        = ScanShader::ScanKernel;


        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        auto   init_key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>());
        auto   ms_tile_state_init_it = ms_scan_key.find(init_key);
        if(ms_tile_state_init_it == ms_scan_key.end())
        {
            auto shader = ScanShader().compile_scan_tile_state_init(m_device);
            if (!shader) { return -1; }
            ms_scan_key.try_emplace(init_key, std::move(shader));
            ms_tile_state_init_it = ms_scan_key.find(init_key);
        }
        if(ms_tile_state_init_it == ms_scan_key.end()) { return -1; }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ScanTileStateInitKernel*>(&(*ms_tile_state_init_it->second));
        if(!ms_scan_tile_state_init_ptr) { return -1; }
        uint init_threads = m_block_size * static_cast<uint>(init_num_blocks);
        auto init_launched = false;
        // ---- interop fast path: tile-state init ----
        // Usage table: `InitializeWardStatus` only writes `tile_status`
        // (head padding + per-tile INVALID); tile_partial / tile_inclusive are
        // never touched by the init kernel.  They are declared READ_WRITE
        // conservatively -- the following scan kernel is ordered behind this
        // dispatch either way (tile_status RAW), so no extra barrier is paid.
        if(m_interop != nullptr)
        {
            auto it = ms_scan_key_interop.find(init_key);
            if(it == ms_scan_key_interop.end())
            {
                it = import_into_interop_cache(
                    m_interop,
                    ms_scan_key_interop,
                    init_key,
                    ScanShader().compile_scan_tile_state_init(m_interop->cuda_device()));
            }
            if(it != ms_scan_key_interop.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    uint>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, uint(num_tiles))
                               .dispatch(init_threads);
                init_launched = true;
            }
        }
        if(!init_launched)
        {
            cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, tile_partial, tile_inclusive, uint(num_tiles))
                           .dispatch(init_threads);
        }

        // scan
        auto key = get_type_and_op_desc<Type4Byte>(scan_op);
        auto &scan_map = is_inclusive ? ms_inclusive_scan_map : ms_exclusive_scan_map;
        auto ms_scan_it = scan_map.find(key);
        if(ms_scan_it == scan_map.end())
        {
            if(is_inclusive)
            {
                auto shader = ScanShader().template compile<true>(m_device, m_shared_mem_size, scan_op);
                if (!shader) { return -1; }
                ms_inclusive_scan_map.try_emplace(key, std::move(shader));
                ms_scan_it = ms_inclusive_scan_map.find(key);
            }
            else
            {
                auto shader = ScanShader().template compile<false>(m_device, m_shared_mem_size, scan_op);
                if (!shader) { return -1; }
                ms_exclusive_scan_map.try_emplace(key, std::move(shader));
                ms_scan_it = ms_exclusive_scan_map.find(key);
            }
        }
        if(ms_scan_it == scan_map.end()) { return -1; }
        auto ms_scan_ptr = reinterpret_cast<ScanShaderKernel*>(&(*ms_scan_it->second));
        if(!ms_scan_ptr) { return -1; }
        uint scan_threads = m_block_size * num_tiles;
        auto scan_launched = false;
        // ---- interop fast path: decoupled-look-back scan ----
        // Usage table: the tile-state triple is the decoupled-look-back channel
        // (volatile_write of partial/inclusive values + status words, then
        // volatile_read/WaitForValid spin-polling by later tiles) so it is
        // READ_WRITE; d_in is read once per element and never written; d_out is
        // written and never read.
        if(m_interop != nullptr)
        {
            auto &scan_interop_map = is_inclusive ? ms_inclusive_scan_interop_map : ms_exclusive_scan_interop_map;
            auto  it               = scan_interop_map.find(key);
            if(it == scan_interop_map.end())
            {
                auto compiled = is_inclusive ?
                                    ScanShader().template compile<true>(
                                        m_interop->cuda_device(), m_shared_mem_size, scan_op) :
                                    ScanShader().template compile<false>(
                                        m_interop->cuda_device(), m_shared_mem_size, scan_op);
                it = import_into_interop_cache(m_interop, scan_interop_map, key, compiled);
            }
            if(it != scan_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::WRITE>,
                    Type4Byte,
                    uint>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, d_in, d_out, initial_value, uint(num_items))
                               .dispatch(scan_threads);
                scan_launched = true;
            }
        }
        if(!scan_launched)
        {
            cmdlist << (*ms_scan_ptr)(tile_states, tile_partial, tile_inclusive, d_in, d_out, initial_value, num_items)
                           .dispatch(scan_threads);
        }
        return 0;
    };


    template <NumericT KeyValue, NumericT ValueType, typename ScanOp, typename FlagValueT = KeyValuePair<int, ValueType>>
    [[nodiscard]] int scan_by_key_array(CommandList&           cmdlist,
                           BufferView<uint>       tile_states,
                           BufferView<FlagValueT> tile_partial,
                           BufferView<FlagValueT> tile_inclusive,
                           BufferView<KeyValue>   d_keys_in,
                           BufferView<KeyValue>   d_prev_keys_in,
                           BufferView<ValueType>  d_values_in,
                           BufferView<ValueType>  d_values_out,
                           size_t                 num_items,
                           ScanOp                 scan_op,
                           ValueType              initial_value,
                           bool                   is_inclusive)
    {
        uint num_tiles = imax(1, ceil_div(num_items, (ITEMS_PER_THREAD * m_block_size)));

        using ScanByKeyShader = details::ScanByKeyModule<KeyValue, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanByKeyTileStateInitKernel = ScanByKeyShader::ScanTileStateInitKernel;
        using ScanByKeyShaderKernel        = ScanByKeyShader::ScanByKeyKernel;

        size_t init_num_blocks                 = ceil_div(num_tiles, m_block_size);
        auto   init_key                        = get_type_and_op_desc<KeyValue, ValueType>();
        auto ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        if(ms_scan_by_key_tile_state_init_it == ms_scan_by_key_tile_state_init_map.end())
        {
            auto shader = ScanByKeyShader().compile_scan_tile_state_init(m_device);
            if (!shader) { return -1; }
            ms_scan_by_key_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        }
        if(ms_scan_by_key_tile_state_init_it == ms_scan_by_key_tile_state_init_map.end()) { return -1; }
        auto ms_scan_by_key_tile_state_init_ptr =
            reinterpret_cast<ScanByKeyTileStateInitKernel*>(&(*ms_scan_by_key_tile_state_init_it->second));
        if(!ms_scan_by_key_tile_state_init_ptr) { return -1; }
        uint init_threads = m_block_size * static_cast<uint>(init_num_blocks);
        auto init_launched = false;
        // ---- interop fast path: tile-state init (+ d_prev_keys_in) ----
        // Usage table: `InitializeWardStatus` writes tile_status, and the same
        // kernel writes the one key preceding each tile into d_prev_keys_in;
        // d_keys_in is read for it.  tile_partial / tile_inclusive are declared
        // READ_WRITE conservatively (see the plain scan path above).
        if(m_interop != nullptr)
        {
            auto it = ms_scan_by_key_tile_state_init_interop_map.find(init_key);
            if(it == ms_scan_by_key_tile_state_init_interop_map.end())
            {
                it = import_into_interop_cache(
                    m_interop,
                    ms_scan_by_key_tile_state_init_interop_map,
                    init_key,
                    ScanByKeyShader().compile_scan_tile_state_init(m_interop->cuda_device()));
            }
            if(it != ms_scan_by_key_tile_state_init_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::WRITE>,
                    int>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, int(num_tiles))
                               .dispatch(init_threads);
                init_launched = true;
            }
        }
        if(!init_launched)
        {
            cmdlist << (*ms_scan_by_key_tile_state_init_ptr)(
                           tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, uint(num_tiles))
                           .dispatch(init_threads);
        }

        // scan
        auto key           = get_type_and_op_desc<KeyValue, ValueType>(scan_op);
        auto &scan_by_key_map = is_inclusive ? ms_inclusive_scan_by_key_map : ms_exclusive_scan_by_key_map;
        auto ms_scan_by_it = scan_by_key_map.find(key);
        if(ms_scan_by_it == scan_by_key_map.end())
        {
            LUISA_INFO("Compiling Scan By Key shader (cache miss)");
            if(is_inclusive)
            {
                auto shader = ScanByKeyShader().template compile<true>(m_device, m_shared_mem_size, scan_op);
                if (!shader) { return -1; }
                ms_inclusive_scan_by_key_map.try_emplace(key, std::move(shader));
                ms_scan_by_it = ms_inclusive_scan_by_key_map.find(key);
            }
            else
            {
                auto shader = ScanByKeyShader().template compile<false>(m_device, m_shared_mem_size, scan_op);
                if (!shader) { return -1; }
                ms_exclusive_scan_by_key_map.try_emplace(key, std::move(shader));
                ms_scan_by_it = ms_exclusive_scan_by_key_map.find(key);
            }
        }
        if(ms_scan_by_it == scan_by_key_map.end()) { return -1; }
        auto ms_scan_by_key_ptr = reinterpret_cast<ScanByKeyShaderKernel*>(&(*ms_scan_by_it->second));
        if(!ms_scan_by_key_ptr) { return -1; }
        uint scan_threads  = m_block_size * num_tiles;
        auto scan_launched = false;
        // ---- interop fast path: decoupled-look-back scan-by-key ----
        // Usage table: same tile-state channel as the plain scan (READ_WRITE),
        // d_keys_in / d_prev_keys_in / d_values_in are read only, d_values_out
        // is written only.
        if(m_interop != nullptr)
        {
            auto &scan_by_key_interop_map =
                is_inclusive ? ms_inclusive_scan_by_key_interop_map : ms_exclusive_scan_by_key_interop_map;
            auto it = scan_by_key_interop_map.find(key);
            if(it == scan_by_key_interop_map.end())
            {
                auto compiled = is_inclusive ?
                                    ScanByKeyShader().template compile<true>(
                                        m_interop->cuda_device(), m_shared_mem_size, scan_op) :
                                    ScanByKeyShader().template compile<false>(
                                        m_interop->cuda_device(), m_shared_mem_size, scan_op);
                it = import_into_interop_cache(m_interop, scan_by_key_interop_map, key, compiled);
            }
            if(it != scan_by_key_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::WRITE>,
                    ValueType,
                    uint>();
                cmdlist << kernel(tile_states,
                                  tile_partial,
                                  tile_inclusive,
                                  d_keys_in,
                                  d_prev_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  initial_value,
                                  uint(num_items))
                               .dispatch(scan_threads);
                scan_launched = true;
            }
        }
        if(!scan_launched)
        {
            cmdlist << (*ms_scan_by_key_ptr)(
                           tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, d_values_in, d_values_out, initial_value, num_items)
                           .dispatch(scan_threads);
        }
        return 0;
    }

    detail::ShaderCacheMap ms_scan_key;
    detail::ShaderCacheMap ms_exclusive_scan_map;
    detail::ShaderCacheMap ms_inclusive_scan_map;

    detail::ShaderCacheMap ms_scan_by_key_tile_state_init_map;
    detail::ShaderCacheMap ms_exclusive_scan_by_key_map;
    detail::ShaderCacheMap ms_inclusive_scan_by_key_map;

    // epoch B: vk-cuda interop kernels shadowing the maps above.  Declared
    // after them so teardown order is the same; they must be destroyed before
    // the vk device the borrowed m_interop points at (risk R6).
    detail::InteropShaderMap ms_scan_key_interop;
    detail::InteropShaderMap ms_exclusive_scan_interop_map;
    detail::InteropShaderMap ms_inclusive_scan_interop_map;
    detail::InteropShaderMap ms_scan_by_key_tile_state_init_interop_map;
    detail::InteropShaderMap ms_exclusive_scan_by_key_interop_map;
    detail::InteropShaderMap ms_inclusive_scan_by_key_interop_map;
};
}  // namespace luisa::parallel_primitive
