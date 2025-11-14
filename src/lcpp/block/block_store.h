/*
 * @Author: Ligo 
 * @Date: 2025-10-14 16:49:47 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:53:54
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


template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::BLOCK_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::BufferVar<T>&                      block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items)
{
    compute::UInt tid         = linear_tid & compute::UInt(WARP_SIZE - 1);
    compute::UInt wid         = linear_tid >> compute::log2(compute::UInt(WARP_SIZE));
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        block_itr.write(thread_offset + (item * compute::UInt(WARP_SIZE)), items[item]);
    }
}

template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::BLOCK_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::BufferVar<T>&                      block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items,
                            compute::UInt                               valid_item)
{
    compute::UInt tid         = linear_tid & compute::UInt(WARP_SIZE - 1);
    compute::UInt wid         = linear_tid >> compute::log2(compute::UInt(WARP_SIZE));
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        $if(warp_offset + tid + (item * compute::UInt(WARP_SIZE)) < valid_item)
        {
            block_itr.write(thread_offset + (item * compute::UInt(WARP_SIZE)), items[item]);
        };
    }
}


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
               const compute::BufferVar<Type4Byte>&                  d_out,
               compute::UInt                                         block_item_start)
    {
        Store(thread_data, d_out, block_item_start, compute::UInt(BlockSize * ITEMS_PER_THREAD));
    }

    void Store(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
               const compute::BufferVar<Type4Byte>&                  d_out,
               compute::UInt                                         block_item_start,
               compute::UInt                                         block_item_end)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = thread_id().x;

        if(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_DIRECT)
        {
            StoreDirectedBlocked(thid * UInt(ITEMS_PER_THREAD), thread_data, d_out, block_item_start, block_item_end);
        };
    };

  private:
    void StoreDirectedBlocked(compute::UInt                                         linear_tid,
                              const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                              const compute::BufferVar<Type4Byte>&                  d_out,
                              compute::UInt block_item_start,
                              compute::UInt block_item_end)
    {
        using namespace luisa::compute;
        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
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