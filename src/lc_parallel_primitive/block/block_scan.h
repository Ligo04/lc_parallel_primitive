/*
 * @Author: Ligo 
 * @Date: 2025-09-28 15:37:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-16 18:09:52
 */
#pragma once
#include <luisa/dsl/var.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/builtin.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/thread/thread_reduce.h>
#include <lc_parallel_primitive/thread/thread_scan.h>


namespace luisa::parallel_primitive
{

enum class DefaultBlockScanAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};
template <typename Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 2, DefaultBlockScanAlgorithm Algorithm = DefaultBlockScanAlgorithm::SHARED_MEMORY>
class BlockScan : public LuisaModule
{

  public:
    BlockScan()
    {
        if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize};
        };
        m_block_aggregate = new SmemType<Type4Byte>{1};
    }
    ~BlockScan() = default;

  public:
    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       output_block_scan,
                       ScanOp                scan_op,
                       Var<Type4Byte>        initial_value = Type4Byte(0))
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
                        scan_op((*m_shared_mem)[index], (*m_shared_mem)[index - offset]);
                };
                offset <<= 1;
                sync_block();
            };
            $if(thid == 0)
            {
                // clear the last element for exclusive scan
                (*m_shared_mem)[block_size_ - 1] = initial_value;
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
                    (*m_shared_mem)[index] = scan_op((*m_shared_mem)[index], temp);
                };
                offset >>= 1;
                sync_block();
            };

            output_block_scan = (*m_shared_mem)[thid];
        }
        $else{};
    }

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       output_block_scan,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op,
                       Var<Type4Byte>        initial_value = Type4Byte(0))
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
                        scan_op((*m_shared_mem)[index], (*m_shared_mem)[index - offset]);
                };
                offset <<= 1;
                sync_block();
            };
            $if(thid == 0)
            {
                // clear the last element for exclusive scan
                (*m_shared_mem)[block_size_ - 1] = initial_value;
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
                    (*m_shared_mem)[index] = scan_op((*m_shared_mem)[index], temp);
                };
                offset >>= 1;
                sync_block();
            };

            output_block_scan = (*m_shared_mem)[thid];

            $if(thid == Int(BlockSize - 1))
            {
                (*m_block_aggregate)[0] = scan_op(thread_data, output_block_scan);
            };
            sync_block();
            block_aggregate = (*m_block_aggregate)[0];
        }
        $else{};
    }

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       ScanOp         op,
                       Var<Type4Byte> initial_value = Type4Byte(0))
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);

        Var<Type4Byte> thread_aggregate =
            ThreadReduce<Type4Byte>().Reduce<ITEMS_PER_THREAD>(thread_datas, op);

        Var<Type4Byte> thread_output;
        ExclusiveScan(thread_aggregate, thread_output, op, initial_value);

        ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(
            thread_datas, output_block_sums, op, thread_output);
    }

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       Var<Type4Byte>& block_aggregate,
                       ScanOp          op,
                       Var<Type4Byte>  initial_value = Type4Byte(0))
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);

        Var<Type4Byte> thread_aggregate =
            ThreadReduce<Type4Byte>().Reduce<ITEMS_PER_THREAD>(thread_datas, op);
    }

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       output_block_sum,
                       ScanOp                op,
                       Var<Type4Byte>        initial_value = Type4Byte(0))
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
                (*m_shared_mem)[block_size_ - 1] = initial_value;
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

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       output_block_sum,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                op,
                       Var<Type4Byte>        initial_value = Type4Byte(0))
    {
        InclusiveScan(thread_data, output_block_sum, op, initial_value);
        $if(compute::thread_id().x == compute::UInt(BlockSize - 1))
        {
            block_aggregate = output_block_sum;
        };
    }


    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       ScanOp         op,
                       Var<Type4Byte> initial_value = Type4Byte(0))
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
                (*m_shared_mem)[block_size_ - 1] = initial_value;
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

    template <typename ScanOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sums,
                       Var<Type4Byte>& block_aggregate,
                       ScanOp          op,
                       Var<Type4Byte>  initial_value = Type4Byte(0))
    {
        InclusiveScan(thread_datas, output_block_sums, op, initial_value);
        $if(compute::thread_id().x == compute::UInt(BlockSize - 1))
        {
            block_aggregate = output_block_sums[ITEMS_PER_THREAD - 1];
        };
    }

    // sum

    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& output_block_sum)
    {
        return ExclusiveScan(thread_data,
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

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sum,
                      Var<Type4Byte>& block_aggregate)
    {
        return ExclusiveScan(thread_data,
                             output_block_sum,
                             block_aggregate,
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

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sum)
    {
        return InclusiveScan(thread_data,
                             output_block_sum,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                             { return a + b; });
    }

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& output_block_sum,
                      Var<Type4Byte>& block_aggregate)
    {
        return InclusiveScan(thread_data,
                             output_block_sum,
                             block_aggregate,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                             { return a + b; });
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
    SmemTypePtr<Type4Byte> m_block_aggregate;
};
}  // namespace luisa::parallel_primitive