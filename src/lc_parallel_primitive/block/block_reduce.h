/*
 * @Author: Ligo 
 * @Date: 2025-09-28 16:54:51 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 14:22:16
 */

#pragma once

#include "luisa/core/basic_traits.h"
#include "luisa/dsl/var.h"
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <cstddef>

namespace luisa::parallel_primitive
{
enum class DefaultBlockReduceAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

template <NumericT Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 4, DefaultBlockReduceAlgorithm Algorithm = DefaultBlockReduceAlgorithm::SHARED_MEMORY>
class BlockReduce : public LuisaModule
{
  public:
    BlockReduce()
    {
        if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize};
        };
    };
    BlockReduce(SmemTypePtr<Type4Byte>& shared_mem)
        : m_shared_mem(shared_mem) {};
    ~BlockReduce() = default;

  public:
    template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& thread_data, ReduceOp op)
    {
        Var<Type4Byte> result = Type4Byte(0);
        $if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
        {
            using namespace luisa::compute;
            luisa::compute::set_block_size(BlockSize);
            Int thid              = Int(thread_id().x);
            (*m_shared_mem)[thid] = thread_data;
            sync_block();
            UInt stride = BlockSize >> 1;
            $while(stride > 0)
            {
                $if(thid < stride)
                {
                    (*m_shared_mem)[thid] =
                        op((*m_shared_mem)[thid], (*m_shared_mem)[thid + stride]);
                };
                sync_block();
                stride >>= 1;
            };

            $if(thid == 0)
            {
                result = (*m_shared_mem)[0];
            };
        }
        $else
        {
            //TODO: implement block-level reduce using warp shuffle
            using namespace luisa::compute;
            luisa::compute::set_block_size(BlockSize);
            Int thid    = Int(thread_id().x);
            Int warp_id = thid / 32;
            Int lane_id = thid % 32;

            result = thread_data;
        };
        return result;
    };

    Var<Type4Byte> Sum(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in,
                      [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                      { return a + b; });
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in, luisa::compute::max);
    }

    Var<Type4Byte> Min(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in, luisa::compute::min);
    }

    template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                          ReduceOp op)
    {
        Var<Type4Byte> result = Type4Byte(0);

        // first, each thread reduce its own data
        Var<Type4Byte> thread_agg = thread_data[0];
        $for(i, 1u, compute::UInt(ITEMS_PER_THREAD))
        {
            thread_agg = op(thread_agg, thread_data[i]);
        };

        $if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
        {
            using namespace luisa::compute;
            luisa::compute::set_block_size(BlockSize);
            Int thid              = Int(thread_id().x);
            (*m_shared_mem)[thid] = thread_agg;
            sync_block();
            UInt stride = BlockSize >> 1;
            $while(stride > 0)
            {
                $if(thid < stride)
                {
                    (*m_shared_mem)[thid] =
                        op((*m_shared_mem)[thid], (*m_shared_mem)[thid + stride]);
                };
                sync_block();
                stride >>= 1;
            };

            $if(thid == 0)
            {
                result = (*m_shared_mem)[0];
            };
        }
        $else
        {
            //TODO: implement block-level reduce using warp shuffle
            using namespace luisa::compute;
            luisa::compute::set_block_size(BlockSize);
            Int thid    = Int(thread_id().x);
            Int warp_id = thid / 32;
            Int lane_id = thid % 32;
        };
        return result;
    };

    Var<Type4Byte> Sum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in)
    {
        return Reduce(d_in,
                      [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                      { return a + b; });
    }

    Var<Type4Byte> Max(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in)
    {
        return Reduce(d_in, luisa::compute::max);
    }

    Var<Type4Byte> Min(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in)
    {
        return Reduce(d_in, luisa::compute::min);
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive