/*
 * @Author: Ligo 
 * @Date: 2025-09-28 16:54:51 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 14:22:16
 */

#pragma once

#include "luisa/dsl/builtin.h"
#include <lc_parallel_primitive/common/type_trait.h>
#include <luisa/dsl/func.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <cstddef>


namespace luisa::parallel_primitive
{
enum class DefaultBlockReduceAlgorithm
{
    NAIVE,
    SHARED_MEMORY,
    WARP_SHUFFLE
};

template <NumericT Type4Byte, size_t BlockSize = 256, DefaultBlockReduceAlgorithm Algorithm = DefaultBlockReduceAlgorithm::SHARED_MEMORY>
class BlockReduce : public LuisaModule
{
    using ReduceOpCallable = luisa::compute::Callable<Var<Type4Byte>()>;

  public:
    BlockReduce(SmemType<Type4Byte>& temp_buffer)
        : m_shared_mem(temp_buffer)
    {
    }
    ~BlockReduce() = default;

  private:
    template <typename ReduceOp>
    ReduceOpCallable Reduce(const Var<Type4Byte>& d_in, ReduceOp op)
    {
        return [&]()
        {
            using namespace luisa::compute;
            luisa::compute::set_block_size(BlockSize);

            $if(Algorithm == DefaultBlockReduceAlgorithm::SHARED_MEMORY)
            {
                m_shared_mem       = SmemType<Type4Byte>(BlockSize);
                Int thid           = Int(thread_id().x);
                m_shared_mem[thid] = d_in;
                sync_block();

                Type4Byte result = def(0);
                UInt      stride = BlockSize >> 1;
                $while(true)
                {
                    $if(thid < stride)
                    {
                        m_shared_mem[thid] =
                            op(m_shared_mem[thid], m_shared_mem[thid + stride]);
                    };
                    sync_block();
                    stride >>= 1;
                    $if(stride == 0)
                    {
                        $break;
                    };
                };

                $if(thid == 0)
                {
                    result = m_shared_mem[0];
                };
                return result;
            }
            $elif(Algorithm == DefaultBlockReduceAlgorithm::WARP_SHUFFLE){
                //TODO implement block-level reduce using warp shuffle
            };
        };
    };

    ReduceOpCallable Sum(const Var<Type4Byte>& d_in)
    {
        return [&]() { return this->Reduce(d_in, sum_op); };
    }

    ReduceOpCallable Max(const Var<Type4Byte>& d_in)
    {
        return [&]() { return this->Reduce(d_in, max_op); };
    }

    ReduceOpCallable Min(const Var<Type4Byte>& d_in)
    {
        return [&]() { return this->Reduce(d_in, min_op); };
    }


  private:
    SmemType<Type4Byte> m_shared_mem;

    static inline Var<Type4Byte> sum_op(const Var<Type4Byte>& a, const Var<Type4Byte>& b)
    {
        return a + b;
    }

    static inline Var<Type4Byte> max_op(const Var<Type4Byte>& a, const Var<Type4Byte>& b)
    {
        return luisa::compute::max(a, b);
    }

    static inline Var<Type4Byte> min_op(const Var<Type4Byte>& a, const Var<Type4Byte>& b)
    {
        return luisa::compute::min(a, b);
    };
};
}  // namespace luisa::parallel_primitive