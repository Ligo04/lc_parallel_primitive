/*
 * @Author: Ligo 
 * @Date: 2025-09-28 15:37:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:44:03
 */
#pragma once
#include "luisa/ast/type.h"
#include "luisa/dsl/var.h"
#include <luisa/dsl/func.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>


namespace luisa::parallel_primitive
{

enum class DefaultBlockScanAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};
template <NumericT Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 4, DefaultBlockScanAlgorithm Algorithm = DefaultBlockScanAlgorithm::SHARED_MEMORY>
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
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       ScanOp op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);

        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> prefix;
        prefix[0] = Type4Byte(0);
        $for(i, 1u, compute::UInt(ITEMS_PER_THREAD))
        {
            prefix[i] = op(prefix[i - 1], thread_datas[i - 1]);
        };

        Var<Type4Byte> thread_aggregate =
            op(prefix[ITEMS_PER_THREAD - 1], thread_datas[ITEMS_PER_THREAD - 1]);

        $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            Int block_size_ = Int(BlockSize);
            Int thid        = Int(thread_id().x);

            (*m_shared_mem)[thid] = thread_aggregate;
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

            // output result
            Var<Type4Byte> thread_offset = (*m_shared_mem)[thid];
            $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
            {
                output_block_sums[i] = op(prefix[i], thread_offset);
            };
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

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       ScanOp op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);

        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> prefix;
        prefix[0] = thread_datas[0];
        $for(i, 1u, compute::UInt(ITEMS_PER_THREAD))
        {
            prefix[i] = op(prefix[i - 1], thread_datas[i - 1]);
        };

        Var<Type4Byte> thread_aggregate =
            op(prefix[ITEMS_PER_THREAD - 1], thread_datas[ITEMS_PER_THREAD - 1]);

        $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            Int block_size_ = Int(BlockSize);
            Int thid        = Int(thread_id().x);

            (*m_shared_mem)[thid] = thread_aggregate;
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

            // output result
            Var<Type4Byte> thread_offset = (*m_shared_mem)[thid];
            $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
            {
                output_block_sums[i] = op(prefix[i], thread_offset + thread_datas[i]);
            };
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

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sum)
    {
        return ExclusiveScan(thread_data,
                             output_block_sum,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                             { return a + b; });
    }
    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sum)
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