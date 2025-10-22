/*
 * @Author: Ligo 
 * @Date: 2025-09-28 16:54:51 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 17:04:16
 */

#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/detail/block_reduce_warp.h>
#include <lcpp/block/detail/block_reduce_mem.h>
#include <lcpp/thread/thread_reduce.h>
#include <cstddef>

namespace luisa::parallel_primitive
{
enum class DefaultBlockReduceAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

template <NumericT Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 2, size_t WARP_SIZE = 32, DefaultBlockReduceAlgorithm Algorithm = DefaultBlockReduceAlgorithm::WARP_SHUFFLE>
class BlockReduce : public LuisaModule
{
  public:
    BlockReduce()
    {
        if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize};
        }
        else if(Algorithm == DefaultBlockReduceAlgorithm::WARP_SHUFFLE)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize / WARP_SIZE};
        };
    };
    BlockReduce(SmemTypePtr<Type4Byte>& shared_mem)
        : m_shared_mem(shared_mem) {};
    ~BlockReduce() = default;

  public:
    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& thread_data, ReduceOp reduce_op)
    {
        Var<Type4Byte> result;
        if(Algorithm == DefaultBlockReduceAlgorithm::WARP_SHUFFLE)
        {
            result = details::BlockReduceShfl<Type4Byte, BlockSize>().Reduce<true>(
                m_shared_mem, thread_data, reduce_op, compute::block_size().x);
        }
        return result;
    };

    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& thread_data, ReduceOp reduce_op, compute::UInt num_item)
    {
        Var<Type4Byte> result;
        if(Algorithm == DefaultBlockReduceAlgorithm::WARP_SHUFFLE)
        {
            $if(num_item >= compute::block_size().x)
            {
                result = details::BlockReduceShfl<Type4Byte, BlockSize>().Reduce<true>(
                    m_shared_mem, thread_data, reduce_op, num_item);
            }
            $else
            {
                result = details::BlockReduceShfl<Type4Byte, BlockSize>().Reduce<false>(
                    m_shared_mem, thread_data, reduce_op, num_item);
            };
        }
        else if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
        {
            result = details::BlockReduceMem<Type4Byte, BlockSize>().Reduce(
                m_shared_mem, thread_data, reduce_op, num_item);
        };
        return result;
    };

    Var<Type4Byte> Sum(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in,
                      [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
                      { return a + b; });
    }

    Var<Type4Byte> Sum(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(
            d_in,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            num_item);
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in, luisa::compute::max);
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, luisa::compute::max, num_item);
    }


    Var<Type4Byte> Min(const Var<Type4Byte>& d_in)
    {
        return Reduce(d_in, luisa::compute::min);
    }

    Var<Type4Byte> Min(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, luisa::compute::min, num_item);
    }

    template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                          ReduceOp      op,
                          compute::UInt num_item)
    {
        Var<Type4Byte> thread_agg =
            ThreadReduce<Type4Byte>().Reduce<ITEMS_PER_THREAD>(thread_data, op);

        return Reduce(thread_agg, op, num_item);
    };

    Var<Type4Byte> Sum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in,
                       compute::UInt num_item)
    {
        return Reduce(
            d_in,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b)
            { return a + b; },
            num_item);
    }

    Var<Type4Byte> Max(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in,
                       compute::UInt num_item)
    {
        return Reduce(d_in, luisa::compute::max, num_item);
    }

    Var<Type4Byte> Min(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in,
                       compute::UInt num_item)
    {
        return Reduce(d_in, luisa::compute::min, num_item);
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive