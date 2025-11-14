/*
 * @Author: Ligo 
 * @Date: 2025-09-28 15:37:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 18:19:36
 */
#pragma once
#include <luisa/dsl/var.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/thread/thread_scan.h>
#include <lcpp/block/detail/block_scan_warp.h>

namespace luisa::parallel_primitive
{
enum class BlockScanAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};
template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = 2, size_t WARP_SIZE = details::WARP_SIZE, BlockScanAlgorithm DEFALUTE_ALGORITHNM = BlockScanAlgorithm::WARP_SHUFFLE>
class BlockScan : public LuisaModule
{
  public:
    BlockScan()
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem      = new SmemType<Type4Byte>{BLOCK_SIZE};
            m_block_aggregate = new SmemType<Type4Byte>{1};
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            m_shared_mem = new SmemType<Type4Byte>{BLOCK_SIZE / WARP_SIZE};
        };
    }
    ~BlockScan() = default;

  public:
    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output, ScanOp scan_op)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(m_shared_mem, thread_data, exclusive_output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, thread_data, exclusive_output, block_aggregate, scan_op);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {

        Var<Type4Byte> block_aggregate;
        ExclusiveScan(m_shared_mem, thread_data, exclusive_output, block_aggregate, scan_op, initial_value);
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {

        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, thread_data, exclusive_output, block_aggregate, scan_op, initial_value);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_out, ScanOp scan_op, BlockPrefixCallbackOp prefix_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, thread_data, exclusive_out, scan_op, prefix_op);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }


    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       ScanOp                                                scan_op)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(thread_datas, output_block_sums, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(thread_datas, output_block_sums, block_aggregate, scan_op, initial_value);
    }


    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       exclusive_output,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                ExclusiveScan(thread_datas[0], exclusive_output[0], block_aggregate, scan_op);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, block_aggregate, scan_op);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(
                    thread_datas, exclusive_output, scan_op, thread_output, compute::thread_x() != 0u);
            };
        };
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                ExclusiveScan(thread_datas[0], output_block_sums[0], block_aggregate, scan_op, initial_value);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, block_aggregate, scan_op, initial_value);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(
                    thread_datas, output_block_sums, scan_op, thread_output);
            };
        };
    }


    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       ScanOp                                                scan_op,
                       BlockPrefixCallbackOp                                 prefix_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                ExclusiveScan(thread_datas[0], output_block_sums[0], scan_op, prefix_op);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, scan_op, prefix_op);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(
                    thread_datas, output_block_sums, scan_op, thread_output);
            };
        };
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_out, ScanOp scan_op)
    {
        Var<Type4Byte> block_aggregate;
        InclusiveScan(m_shared_mem, thread_data, inclusive_out, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_out,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, thread_data, inclusive_out, block_aggregate, scan_op);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_out,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op,
                       Var<Type4Byte>        initial_value)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, thread_data, inclusive_out, block_aggregate, scan_op, initial_value);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }


    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_out, ScanOp scan_op, BlockPrefixCallbackOp prefix_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, thread_data, inclusive_out, scan_op, prefix_op);
        }
        else if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::SHARED_MEMORY)
        {
        };
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       inclusive_out,
                       ScanOp                                                scan_op)
    {
        Var<Type4Byte> block_aggregate;
        InclusiveScan(thread_datas, inclusive_out, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       inclusive_out,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                InclusiveScan(thread_datas[0], inclusive_out[0], block_aggregate, scan_op);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, block_aggregate, scan_op);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(
                    thread_datas, inclusive_out, scan_op, thread_output, compute::thread_x() != 0u);
            };
        };
    }


    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                InclusiveScan(thread_datas[0], output_block_sums[0], block_aggregate, scan_op, initial_value);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, block_aggregate, scan_op, initial_value);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(
                    thread_datas, output_block_sums, scan_op, thread_output, compute::thread_x() != 0u);
            };
        };
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_datas,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output_block_sums,
                       ScanOp                                                scan_op,
                       BlockPrefixCallbackOp                                 prefix_op)
    {
        if(DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE)
        {
            if(ITEMS_PER_THREAD == 1)
            {
                InclusiveScan(thread_datas[0], output_block_sums[0], scan_op, prefix_op);
            }
            else
            {
                Var<Type4Byte> thread_aggregate =
                    ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_datas, scan_op);

                Var<Type4Byte> thread_output;
                ExclusiveScan(thread_aggregate, thread_output, scan_op, prefix_op);

                ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(
                    thread_datas, output_block_sums, scan_op, thread_output);
            };
        };
    }

    // sum
    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_out)
    {
        return ExclusiveScan(
            thread_data,
            exclusive_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Var<Type4Byte>(0));
    }
    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_out, Var<Type4Byte>& block_aggregate)
    {
        return ExclusiveScan(
            thread_data,
            exclusive_out,
            block_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Var<Type4Byte>(0));
    }

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       exclusive_out)
    {
        return ExclusiveScan(
            thread_data,
            exclusive_out,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Var<Type4Byte>(0));
    }

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       exclusive_out,
                      Var<Type4Byte>&                                       block_aggregate)
    {
        return ExclusiveScan(
            thread_data,
            exclusive_out,
            block_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Var<Type4Byte>(0));
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_out)
    {
        return InclusiveScan(thread_data,
                             inclusive_out,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; });
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_out, Var<Type4Byte>& block_aggregate)
    {
        return InclusiveScan(thread_data,
                             inclusive_out,
                             block_aggregate,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; });
    }

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       inclusive_out)
    {
        return InclusiveScan(thread_data,
                             inclusive_out,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; });
    }

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       inclusive_out,
                      Var<Type4Byte>&                                       block_aggregate)
    {
        return InclusiveScan(thread_data,
                             inclusive_out,
                             block_aggregate,
                             [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; });
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
    SmemTypePtr<Type4Byte> m_block_aggregate;
};
}  // namespace luisa::parallel_primitive