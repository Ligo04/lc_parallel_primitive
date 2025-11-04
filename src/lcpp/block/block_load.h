/*
 * @Author: Ligo 
 * @Date: 2025-10-14 14:01:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:53:31
 */
#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/sugar.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{

enum class BlockLoadAlgorithm
{
    BLOCK_LOAD_DIRECT    = 0,
    BLOCK_LOAD_TRANSPOSE = 1
};

template <typename Type4Byte, size_t BlockSize = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = 2, BlockLoadAlgorithm DefaultLoadAlgorithm = BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>
class BlockLoad : public LuisaModule
{
  public:
    BlockLoad(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }
    BlockLoad() {}

    ~BlockLoad() = default;

  public:
    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start)
    {
        Load(d_in,
             thread_data,
             block_item_start,
             compute::UInt(BlockSize * ITEMS_PER_THREAD),
             Type4Byte(0));
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start,
              compute::UInt                                   block_item_end)
    {
        Load(d_in, thread_data, block_item_start, block_item_end, Type4Byte(0));
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start,
              compute::UInt                                   block_item_end,
              Var<Type4Byte>                                  default_value)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = thread_id().x;

        if(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_DIRECT)
        {
            LoadDirectedBlocked(thid * UInt(ITEMS_PER_THREAD),
                                d_in,
                                thread_data,
                                block_item_start,
                                block_item_end,
                                default_value);
        };
    }


  private:
    void LoadDirectedBlocked(compute::UInt                        linear_tid,
                             const compute::BufferVar<Type4Byte>& d_in,
                             compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                             compute::UInt  block_item_start,
                             compute::UInt  block_item_end,
                             Var<Type4Byte> default_value)
    {
        using namespace luisa::compute;
        // $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            UInt index = linear_tid + i;
            $if(index < block_item_end)
            {
                thread_data[i] = d_in.read(block_item_start + index);
            }
            $else
            {
                thread_data[i] = default_value;
            };
        };
    }

    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive