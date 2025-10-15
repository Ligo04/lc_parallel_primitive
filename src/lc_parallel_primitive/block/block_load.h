/*
 * @Author: Ligo 
 * @Date: 2025-10-14 14:01:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-15 10:57:10
 */
#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/sugar.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>

namespace luisa::parallel_primitive
{

enum class BlockLoadAlgorithm
{
    BLOCK_LOAD_DIRECT    = 0,
    BLOCK_LOAD_TRANSPOSE = 1
};

template <typename Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 2, BlockLoadAlgorithm DefaultLoadAlgorithm = BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>
class BlockLoad : public LuisaModule
{
  public:
    BlockLoad(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    ~BlockLoad() = default;

  public:
    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data)
    {
        Load(d_in, thread_data, UInt(BlockSize * ITEMS_PER_THREAD), Type4Byte(0));
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_end)
    {
        Load(d_in, thread_data, block_item_end, Type4Byte(0));
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_end,
              Type4Byte                                       default_value)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt block_start = block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);
        UInt thid = thread_id().x;

        $if(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_DIRECT)
        {
            LoadDirectedBlocked(block_start + thid * UInt(ITEMS_PER_THREAD),
                                d_in,
                                thread_data,
                                block_item_end,
                                default_value);
        };
    }


  private:
    void LoadDirectedBlocked(compute::UInt                        linear_tid,
                             const compute::BufferVar<Type4Byte>& d_in,
                             compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                             compute::UInt  block_item_end,
                             Var<Type4Byte> default_value)
    {
        using namespace luisa::compute;
        $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        {
            UInt index = linear_tid * UInt(ITEMS_PER_THREAD) + i;
            $if(index < block_item_end)
            {
                thread_data[i] = d_in.read(index);
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