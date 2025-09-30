/*
 * @Author: Ligo 
 * @Date: 2025-09-28 15:37:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:44:03
 */
#pragma once

#include <lc_parallel_primitive/common/type_trait.h>
#include <luisa/dsl/func.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>


namespace luisa::parallel_primitive
{

enum class DefaultBlockScanAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};
template <NumericT Type4Byte, size_t BlockSize = 256, DefaultBlockScanAlgorithm Algorithm = DefaultBlockScanAlgorithm::SHARED_MEMORY>
class BlockScan : public LuisaModule
{

  public:
    BlockScan()
    {
        if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize};
        };
    }
    ~BlockScan() = default;

  public:
    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void ExclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& output_block_sum, ScanOp op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            Int block_size_ = Int(BlockSize);
            Int thid        = Int(thread_id().x);
            Int global_id   = Int(block_id().x * block_size_x() + thid);

            (*m_shared_mem)[thid] = thread_data;
            sync_block();

            // up-sweep
            UInt offset = def(1);
            $while(offset < block_size_)
            {
                UInt index = (thid + 1) * offset * 2 - 1;
                $if(index < block_size_)
                {
                    (*m_shared_mem)[index] =
                        op((*m_shared_mem)[index], (*m_shared_mem)[index - offset]);
                };
                offset <<= 1;
                sync_block();
            };
            $if(thid == 0)
            {
                // clear the last element for exclusive scan
                (*m_shared_mem)[block_size_ - 1] = 0;
            };
            sync_block();
            // down-sweep
            offset = def(block_size_ >> 1);
            $while(offset > 0)
            {
                UInt index = (thid + 1) * offset * 2 - 1;
                $if(index < block_size_)
                {
                    // swap
                    auto temp = (*m_shared_mem)[index - offset];
                    (*m_shared_mem)[index - offset] = (*m_shared_mem)[index];
                    (*m_shared_mem)[index] = op((*m_shared_mem)[index], temp);
                };
                offset >>= 1;
                sync_block();
            };

            output_block_sum = (*m_shared_mem)[thid];
        }
        $else{};
    }
    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void InclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& output_block_sum, ScanOp op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            Int block_size_ = Int(BlockSize);
            Int thid        = Int(thread_id().x);
            Int global_id   = Int(block_id().x * block_size_x() + thid);

            (*m_shared_mem)[thid] = thread_data;
            sync_block();

            // up-sweep
            UInt offset = def(1);
            $while(offset < block_size_)
            {
                UInt index = (thid + 1) * offset * 2 - 1;
                $if(index < block_size_)
                {
                    (*m_shared_mem)[index] =
                        op((*m_shared_mem)[index], (*m_shared_mem)[index - offset]);
                };
                offset <<= 1;
                sync_block();
            };
            $if(thid == 0)
            {
                // clear the last element for exclusive scan
                (*m_shared_mem)[block_size_ - 1] = 0;
            };
            sync_block();
            // down-sweep
            offset = def(block_size_ >> 1);
            $while(offset > 0)
            {
                UInt index = (thid + 1) * offset * 2 - 1;
                $if(index < block_size_)
                {
                    // swap
                    auto temp = (*m_shared_mem)[index - offset];
                    (*m_shared_mem)[index - offset] = (*m_shared_mem)[index];
                    (*m_shared_mem)[index] = op((*m_shared_mem)[index], temp);
                };
                offset >>= 1;
                sync_block();
            };

            output_block_sum = op((*m_shared_mem)[thid], thread_data);
        }
        $else{};
    }

    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& output_block_sum)
    {
        return ExclusiveScan(thread_data,
                             output_block_sum,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                             { return a + b; });
    }
    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& output_block_sum)
    {
        return InclusiveScan(thread_data,
                             output_block_sum,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                             { return a + b; });
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive