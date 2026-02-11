/*
 * @Author: Ligo 
 * @Date: 2025-10-09 09:52:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:12:25
 */


#pragma once
#include "luisa/runtime/buffer.h"
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
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceScan : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

  public:
    DeviceScan()  = default;
    ~DeviceScan() = default;

    void create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }


    template <NumericT Type4Byte, typename ScanOp>
    void ExclusiveScan(CommandList&          cmdlist,
                       Stream&               stream,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        // tilestate
        using ScanShaderT    = details::ScanModule<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanTileStateT = ScanShaderT::TileState;
        Buffer<ScanTileStateT> tile_states =
            m_device.create_buffer<ScanTileStateT>(details::WARP_SIZE + num_tiles);
        scan_array<Type4Byte>(cmdlist, tile_states.view(), d_in, d_out, num_items, scan_op, initial_value, false);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte, typename ScanOp>
    void InclusiveScan(CommandList&          cmdlist,
                       Stream&               stream,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        // tilestate
        using ScanShaderT    = details::ScanModule<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanTileStateT = ScanShaderT::TileState;
        Buffer<ScanTileStateT> tile_states =
            m_device.create_buffer<ScanTileStateT>(details::WARP_SIZE + num_tiles);
        scan_array<Type4Byte>(cmdlist, tile_states.view(), d_in, d_out, num_items, scan_op, initial_value, true);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT Type4Byte>
    void ExclusiveSum(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_items)
    {
        ExclusiveScan(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_items,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void InclusiveSum(CommandList& cmdlist, Stream& stream, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_items)
    {
        InclusiveScan(
            cmdlist,
            stream,
            d_in,
            d_out,
            num_items,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void ExclusiveScanByKey(CommandList&          cmdlist,
                            Stream&               stream,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        // tilestate
        using ScanByKey = details::ScanByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanByKeyTileState = ScanByKey::ScanTileState;
        Buffer<ScanByKeyTileState> tile_states =
            m_device.create_buffer<ScanByKeyTileState>(details::WARP_SIZE + num_tiles);

        Buffer<KeyType> d_prev_keys_in = m_device.create_buffer<KeyType>(num_tiles);
        scan_by_key_array<KeyType, ValueType>(
            cmdlist, tile_states.view(), d_keys_in, d_prev_keys_in.view(), d_values_in, d_values_out, num_items, scan_op, initial_value, false);
        stream << cmdlist.commit() << synchronize();
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void InclusiveScanByKey(CommandList&          cmdlist,
                            Stream&               stream,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        int num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));
        // tilestate
        using ScanByKey = details::ScanByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanByKeyTileState = ScanByKey::ScanTileState;
        Buffer<ScanByKeyTileState> tile_states =
            m_device.create_buffer<ScanByKeyTileState>(details::WARP_SIZE + num_tiles);

        Buffer<KeyType> d_prev_keys_in = m_device.create_buffer<KeyType>(num_tiles);
        scan_by_key_array<KeyType, ValueType>(
            cmdlist, tile_states.view(), d_keys_in, d_prev_keys_in.view(), d_values_in, d_values_out, num_items, scan_op, initial_value, true);
        stream << cmdlist.commit() << synchronize();
    }


    template <NumericT Type4Byte>
    void ExclusiveSumByKey(CommandList&          cmdlist,
                           Stream&               stream,
                           BufferView<Type4Byte> d_keys_in,
                           BufferView<Type4Byte> d_values_in,
                           BufferView<Type4Byte> d_values_out,
                           size_t                num_items)
    {
        ExclusiveScanByKey(
            cmdlist,
            stream,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            num_items,
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void InclusiveSumByKey(CommandList&          cmdlist,
                           Stream&               stream,
                           BufferView<Type4Byte> d_keys_in,
                           BufferView<Type4Byte> d_values_in,
                           BufferView<Type4Byte> d_values_out,
                           size_t                num_items)
    {
        InclusiveScanByKey(
            cmdlist,
            stream,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            num_items,
            Type4Byte(0));
    }


  private:
    template <NumericT Type4Byte, typename ScanTileStateT, typename ScanOp>
    void scan_array(CommandList&               cmdlist,
                    BufferView<ScanTileStateT> tile_states,
                    BufferView<Type4Byte>      d_in,
                    BufferView<Type4Byte>      d_out,
                    size_t                     num_items,
                    ScanOp                     scan_op,
                    Type4Byte                  initial_value,
                    bool                       is_inclusive)
    {
        auto num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));

        using ScanShader    = details::ScanModule<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanTileState = ScanShader::TileState;
        using ScanTileStateInitKernel = ScanShader::ScanTileStateInitKernel;
        using ScanShaderKernel        = ScanShader::ScanKernel;

        size_t init_num_blocks = ceil(float(num_tiles) / BLOCK_SIZE);
        auto   init_key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto   ms_tile_state_init_it = ms_scan_key.find(init_key);
        if(ms_tile_state_init_it == ms_scan_key.end())
        {
            auto shader = ScanShader().compile_scan_tile_state_init(m_device);
            ms_scan_key.try_emplace(init_key, std::move(shader));
            ms_tile_state_init_it = ms_scan_key.find(init_key);
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ScanTileStateInitKernel*>(&(*ms_tile_state_init_it->second));

        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, uint(num_tiles)).dispatch(init_num_blocks * BLOCK_SIZE);

        // scan
        auto key = get_type_and_op_desc<Type4Byte>(scan_op);
        auto ms_scan_it = is_inclusive ? ms_inclusive_scan_map.find(key) : ms_exclusive_scan_map.find(key);
        if(ms_scan_it == (is_inclusive ? ms_inclusive_scan_map : ms_exclusive_scan_map).end())
        {
            LUISA_INFO("Compiling Scan shader for key: {}", key);
            if(is_inclusive)
            {
                auto shader = ScanShader().template compile<true>(m_device, m_shared_mem_size, scan_op);
                ms_inclusive_scan_map.try_emplace(key, std::move(shader));
                ms_scan_it = ms_inclusive_scan_map.find(key);
            }
            else
            {
                auto shader = ScanShader().template compile<false>(m_device, m_shared_mem_size, scan_op);
                ms_exclusive_scan_map.try_emplace(key, std::move(shader));
                ms_scan_it = ms_exclusive_scan_map.find(key);
            }
        }
        auto ms_scan_ptr = reinterpret_cast<ScanShaderKernel*>(&(*ms_scan_it->second));
        cmdlist << (*ms_scan_ptr)(tile_states, d_in, d_out, initial_value, num_items).dispatch(m_block_size * num_tiles);
    };


    template <NumericT KeyValue, NumericT ValueType, typename ScanTileStateT, typename ScanOp>
    void scan_by_key_array(CommandList&               cmdlist,
                           BufferView<ScanTileStateT> tile_states,
                           BufferView<KeyValue>       d_keys_in,
                           BufferView<KeyValue>       d_prev_keys_in,
                           BufferView<ValueType>      d_values_in,
                           BufferView<ValueType>      d_values_out,
                           size_t                     num_items,
                           ScanOp                     scan_op,
                           ValueType                  initial_value,
                           bool                       is_inclusive)
    {
        auto num_tiles = imax(1, (int)ceil((float)num_items / (ITEMS_PER_THREAD * m_block_size)));

        using ScanByKeyShader = details::ScanByKeyModule<KeyValue, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanByKeyTileState           = ScanByKeyShader::ScanTileState;
        using ScanByKeyTileStateInitKernel = ScanByKeyShader::ScanTileStateInitKernel;
        using ScanByKeyShaderKernel        = ScanByKeyShader::ScanByKeyKernel;

        size_t init_num_blocks                 = ceil(float(num_tiles) / BLOCK_SIZE);
        auto   init_key                        = get_type_and_op_desc<KeyValue, ValueType>();
        auto ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        if(ms_scan_by_key_tile_state_init_it == ms_scan_by_key_tile_state_init_map.end())
        {
            auto shader = ScanByKeyShader().compile_scan_tile_state_init(m_device);
            ms_scan_by_key_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        }
        auto ms_scan_by_key_tile_state_init_ptr =
            reinterpret_cast<ScanByKeyTileStateInitKernel*>(&(*ms_scan_by_key_tile_state_init_it->second));
        cmdlist << (*ms_scan_by_key_tile_state_init_ptr)(tile_states, d_keys_in, d_prev_keys_in, uint(num_tiles))
                       .dispatch(m_block_size * num_tiles);

        // scan
        auto key           = get_type_and_op_desc<KeyValue, ValueType>(scan_op);
        auto ms_scan_by_it = is_inclusive ? ms_inclusive_scan_by_key_map.find(key) :
                                            ms_exclusive_scan_by_key_map.find(key);
        if(ms_scan_by_it
           == (is_inclusive ? ms_inclusive_scan_by_key_map : ms_exclusive_scan_by_key_map).end())
        {
            LUISA_INFO("Compiling Scan By Key shader for key: {}", key);
            if(is_inclusive)
            {
                auto shader = ScanByKeyShader().template compile<true>(m_device, m_shared_mem_size, scan_op);
                ms_inclusive_scan_by_key_map.try_emplace(key, std::move(shader));
                ms_scan_by_it = ms_inclusive_scan_by_key_map.find(key);
            }
            else
            {
                auto shader = ScanByKeyShader().template compile<false>(m_device, m_shared_mem_size, scan_op);
                ms_exclusive_scan_by_key_map.try_emplace(key, std::move(shader));
                ms_scan_by_it = ms_exclusive_scan_by_key_map.find(key);
            }
        }
        auto ms_scan_by_key_ptr = reinterpret_cast<ScanByKeyShaderKernel*>(&(*ms_scan_by_it->second));
        cmdlist << (*ms_scan_by_key_ptr)(tile_states, d_keys_in, d_prev_keys_in, d_values_in, d_values_out, initial_value, num_items)
                       .dispatch(m_block_size * num_tiles);
    }

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_key;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_exclusive_scan_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_inclusive_scan_map;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_by_key_tile_state_init_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_exclusive_scan_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_inclusive_scan_by_key_map;
};
}  // namespace luisa::parallel_primitive