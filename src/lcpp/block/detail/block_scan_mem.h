/*
 * @Author: Ligo 
 * @Date: 2025-10-20 13:51:43 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-20 15:06:27
 */

#pragma once
#include "lcpp/runtime/core.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>

namespace luisa::parallel_primitive
{
namespace details
{
    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    using namespace luisa::compute;
    template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE>
    struct BlockScanMem
    {
        template <typename ScanOp>
        void ExclusiveScan(SmemTypePtr<Type4Byte>& m_shared_mem,
                           const Var<Type4Byte>&   thread_data,
                           Var<Type4Byte>&         exclusive_output,
                           Var<Type4Byte>&         block_aggregate,
                           ScanOp                  scan_op,
                           const Var<Type4Byte>&   initial_value)
        {
            UInt block_size_ = UInt(BLOCK_SIZE);
            UInt thid        = thread_id().x;
            UInt global_id   = block_id().x * block_size_x() + thid;

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

            exclusive_output = (*m_shared_mem)[thid];
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive