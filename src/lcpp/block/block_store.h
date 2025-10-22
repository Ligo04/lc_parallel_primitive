/*
 * @Author: Ligo 
 * @Date: 2025-10-14 16:49:47 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 14:48:57
 */


#pragma once
#include <luisa/dsl/var.h>
#include <luisa/dsl/sugar.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
enum class BlockStoreAlgorithm
{
    BLOCK_STORE_DIRECT    = 0,
    BLOCK_STORE_TRANSPOSE = 1
};

template <typename Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 2, BlockStoreAlgorithm DefaultStoreAlgorithm = BlockStoreAlgorithm::BLOCK_STORE_DIRECT>
class BlockStore : public LuisaModule
{
  public:
    BlockStore(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }
    BlockStore() {}

    ~BlockStore() = default;

  public:
    void Store(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
               const compute::BufferVar<Type4Byte>& d_out,
               compute::UInt                        block_item_start)
    {
        Store(thread_data, d_out, block_item_start, compute::UInt(BlockSize * ITEMS_PER_THREAD));
    }

    void Store(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
               const compute::BufferVar<Type4Byte>& d_out,
               compute::UInt                        block_item_start,
               compute::UInt                        block_item_end)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = thread_id().x;

        $if(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_DIRECT)
        {
            StoreDirectedBlocked(thid * UInt(ITEMS_PER_THREAD), thread_data, d_out, block_item_start, block_item_end);
        };
    };

  private:
    void StoreDirectedBlocked(compute::UInt linear_tid,
                              const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                              const compute::BufferVar<Type4Byte>& d_out,
                              compute::UInt block_item_start,
                              compute::UInt block_item_end)
    {
        using namespace luisa::compute;
        $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        {
            UInt index = linear_tid + i;
            $if(index < block_item_end)
            {
                d_out.write(block_item_start + index, thread_data[i]);
            };
        };
    };

    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive