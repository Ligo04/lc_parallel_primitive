/*
 * @Author: Ligo 
 * @Date: 2025-09-28 16:54:51 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-10 17:06:13
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
#include <lcpp/common/thread_operators.h>
#include <cstddef>

namespace luisa::parallel_primitive
{
enum class BlockReduceAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

template <typename Type4Byte,
          size_t               BLOCK_SIZE       = details::BLOCK_SIZE,
          size_t               ITEMS_PER_THREAD = details::ITEMS_PER_THREAD,
          size_t               WARP_SIZE        = details::WARP_SIZE,
          BlockReduceAlgorithm Algorithm        = BlockReduceAlgorithm::WARP_SHUFFLE>
class BlockReduce : public LuisaModule
{
  public:
    BlockReduce()
    {
        if(Algorithm == BlockReduceAlgorithm::SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{BLOCK_SIZE};
        }
        else if(Algorithm == BlockReduceAlgorithm::WARP_SHUFFLE)
        {
            m_shared_mem = new SmemType<Type4Byte>{BLOCK_SIZE / WARP_SIZE};
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
        if(Algorithm == BlockReduceAlgorithm::WARP_SHUFFLE)
        {
            result = details::BlockReduceShfl<Type4Byte, BLOCK_SIZE>().template Reduce<true>(
                m_shared_mem, thread_data, reduce_op, compute::block_size().x);
        }
        return result;
    };

    template <typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte>& thread_data, ReduceOp reduce_op, compute::UInt num_item)
    {
        Var<Type4Byte> result;
        if(Algorithm == BlockReduceAlgorithm::WARP_SHUFFLE)
        {
            $if(num_item >= compute::block_size().x)
            {
                result = details::BlockReduceShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().template Reduce<true>(
                    m_shared_mem, thread_data, reduce_op, num_item);
            }
            $else
            {
                result = details::BlockReduceShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().template Reduce<false>(
                    m_shared_mem, thread_data, reduce_op, num_item);
            };
        }
        else if(Algorithm == BlockReduceAlgorithm::SHARED_MEMORY)
        {
            result = details::BlockReduceMem<Type4Byte, BLOCK_SIZE>().Reduce(
                m_shared_mem, thread_data, reduce_op, num_item);
        };
        return result;
    };

    Var<Type4Byte> Sum(const Var<Type4Byte>& d_in) { return Reduce(d_in, SumOp()); }

    Var<Type4Byte> Sum(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, SumOp(), num_item);
    }

    Var<Type4Byte> Max(const Var<Type4Byte>& d_in) { return Reduce(d_in, MaxOp()); }

    Var<Type4Byte> Max(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, MaxOp(), num_item);
    }

    Var<Type4Byte> Min(const Var<Type4Byte>& d_in) { return Reduce(d_in, MinOp()); }

    Var<Type4Byte> Min(const Var<Type4Byte>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, MinOp(), num_item);
    }

    template <typename ReduceOp = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, const Var<Type4Byte>&)>>
    Var<Type4Byte> Reduce(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
                          ReduceOp                                              op,
                          compute::UInt                                         num_item)
    {
        Var<Type4Byte> thread_agg = ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(thread_data, op);
        return Reduce(thread_agg, op, num_item);
    };

    Var<Type4Byte> Sum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, SumOp(), num_item);
    }

    Var<Type4Byte> Max(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, MaxOp(), num_item);
    }

    Var<Type4Byte> Min(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& d_in, compute::UInt num_item)
    {
        return Reduce(d_in, MinOp(), num_item);
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive