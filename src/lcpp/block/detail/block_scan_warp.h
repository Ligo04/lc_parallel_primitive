/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 17:49:28
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/warp/warp_scan.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename T>
    using SmemType = luisa::compute::Shared<T>;
    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE>
    struct BlockScanShfl
    {
        template <typename ScanOp>
        void ExclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         exclusive_output,
                           Var<Type4Byte>&         block_aggregate,
                           ScanOp                  scan_op)
        {
            Var<Type4Byte> inclusive_output;
            WarpScan<Type4Byte, WARP_SIZE, WarpScanAlgorithm::WARP_SHUFFLE>().Scan(
                thread_data, inclusive_output, exclusive_output, scan_op);

            Var<Type4Byte> warp_prefix =
                ComputeWarpPrefix(m_shared_mem, scan_op, inclusive_output, block_aggregate);

            UInt warp_id = thread_id().x / warp_lane_count();
            UInt lane_id = warp_lane_id();
            $if(warp_id != 0)
            {
                exclusive_output = scan_op(warp_prefix, exclusive_output);
                $if(lane_id == 0)
                {
                    exclusive_output = warp_prefix;
                };
            };
        }


        template <typename ScanOp>
        void ExclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         exclusive_output,
                           Var<Type4Byte>&         block_aggregate,
                           ScanOp                  scan_op,
                           const Var<Type4Byte>&   initial_value)
        {
            Var<Type4Byte> inclusive_output;
            WarpScan<Type4Byte, WARP_SIZE, WarpScanAlgorithm::WARP_SHUFFLE>().Scan(
                thread_data, inclusive_output, exclusive_output, scan_op, initial_value);

            Var<Type4Byte> warp_prefix =
                ComputeWarpPrefix(m_shared_mem, scan_op, inclusive_output, block_aggregate, initial_value);

            UInt warp_id = thread_id().x / warp_lane_count();
            UInt lane_id = warp_lane_id();

            $if(warp_id != 0)
            {
                exclusive_output = scan_op(exclusive_output, warp_prefix);
                $if(lane_id == 0)
                {
                    exclusive_output = warp_prefix;
                };
            };
        }

        template <typename ScanOp, typename BlockPrefixCallbackT>
        void ExclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         exclusive_output,
                           ScanOp                  scan_op,
                           BlockPrefixCallbackT    block_prefix_callback_op)
        {
            SmemTypePtr<Type4Byte> shared_mem_block_prefix = new SmemType<Type4Byte>{1};

            Var<Type4Byte> block_aggregate;
            ExclusiveScan(m_shared_mem, thread_data, exclusive_output, block_aggregate, scan_op);

            UInt warp_id = thread_id().x / warp_lane_count();
            UInt lane_id = warp_lane_id();
            $if(warp_id == 0)
            {
                Var<Type4Byte> block_prefix = block_prefix_callback_op(block_aggregate);

                $if(lane_id == 0)
                {
                    (*shared_mem_block_prefix)[0] = block_prefix;
                    exclusive_output              = block_prefix;
                };
            };

            sync_block();

            Var<Type4Byte> block_prefix = (*shared_mem_block_prefix)[0];
            $if(thread_id().x > 0)
            {
                exclusive_output = scan_op(block_prefix, exclusive_output);
            };
        }

        template <typename ScanOp>
        void InclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         inclusive_output,
                           Var<Type4Byte>&         block_aggregate,
                           ScanOp                  scan_op)
        {
            WarpScan<Type4Byte, WARP_SIZE>().InclusiveScan(thread_data, inclusive_output, scan_op);

            Var<Type4Byte> warp_prefix =
                ComputeWarpPrefix(m_shared_mem, scan_op, inclusive_output, block_aggregate);

            UInt warp_id = thread_id().x / warp_lane_count();
            $if(warp_id != 0)
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            };
        }

        template <typename ScanOp>
        void InclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         inclusive_output,
                           Var<Type4Byte>&         block_aggregate,
                           ScanOp                  scan_op,
                           const Var<Type4Byte>&   initial_value)
        {
            WarpScan<Type4Byte, WARP_SIZE>().InclusiveScan(thread_data, inclusive_output, scan_op);

            Var<Type4Byte> warp_prefix =
                ComputeWarpPrefix(m_shared_mem, scan_op, inclusive_output, block_aggregate, initial_value);

            UInt warp_id = thread_id().x / warp_lane_count();
            $if(warp_id != 0)
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            };
        }

        template <typename ScanOp, typename BlockPrefixCallbackOp>
        void InclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         inclusive_output,
                           ScanOp                  scan_op,
                           BlockPrefixCallbackOp   prefix_op)
        {
            SmemTypePtr<Type4Byte> shared_mem_block_prefix = new SmemType<Type4Byte>{1};
            Var<Type4Byte>         block_aggregate;
            InclusiveScan(m_shared_mem, thread_data, inclusive_output, block_aggregate, scan_op);

            UInt warp_id = thread_id().x / warp_lane_count();
            $if(warp_id == 0)
            {
                Var<Type4Byte> block_prefix = prefix_op(block_aggregate);
                UInt           lane_id      = warp_lane_id();
                $if(lane_id == 0)
                {
                    (*shared_mem_block_prefix)[0] = block_prefix;
                };
            };

            sync_block();

            Var<Type4Byte> block_prefix = (*shared_mem_block_prefix)[0];
            inclusive_output            = scan_op(block_prefix, inclusive_output);
        }

      private:
        template <typename ScanOp>
        Var<Type4Byte> ComputeWarpPrefix(SmemTypePtr<Type4Byte>& m_shared_mem,
                                         ScanOp                  scan_op,
                                         const Var<Type4Byte>&   warp_aggregate,
                                         Var<Type4Byte>&         block_aggregate)
        {
            UInt warp_id = thread_id().x / warp_lane_count();
            UInt lane_id = warp_lane_id();

            // write warp result to shared memory
            $if(lane_id == warp_lane_count() - 1)
            {
                (*m_shared_mem)[warp_id] = warp_aggregate;
            };

            sync_block();

            Var<Type4Byte> warp_prefix;
            block_aggregate = (*m_shared_mem)[0];

            for(auto item = 1u; item < BLOCK_SIZE / WARP_SIZE; ++item)
            {
                $if(warp_id == item)
                {
                    warp_prefix = block_aggregate;
                };
                Var<Type4Byte> addend = (*m_shared_mem)[item];
                block_aggregate       = scan_op(block_aggregate, addend);
            };
            return warp_prefix;
        }

        template <typename ScanOp>
        Var<Type4Byte> ComputeWarpPrefix(SmemTypePtr<Type4Byte>& m_shared_mem,
                                         ScanOp                  scan_op,
                                         const Var<Type4Byte>&   warp_aggregate,
                                         Var<Type4Byte>&         block_aggregate,
                                         const Var<Type4Byte>&   initial_value)
        {
            Var<Type4Byte> warp_prefix =
                ComputeWarpPrefix(m_shared_mem, scan_op, warp_aggregate, block_aggregate);

            warp_prefix = scan_op(initial_value, warp_prefix);

            UInt warp_id = thread_id().x / warp_lane_count();
            $if(warp_id == 0)
            {
                warp_prefix = initial_value;
            };

            return warp_prefix;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive