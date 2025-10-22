/*
 * @Author: Ligo 
 * @Date: 2025-10-14 14:03:06 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:58:51
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
template <NumericT Type4Byte, size_t BlockSize = 256, size_t ITEMS_PER_THREAD = 2>
class BlockDiscontinuity : public LuisaModule
{
  public:
    struct tempStorage
    {
        SmemTypePtr<Type4Byte> first_element;
        SmemTypePtr<Type4Byte> last_element;
    };

    static compute::Bool Default_flag_op(const Var<Type4Byte>& a, const Var<Type4Byte>& b)
    {
        return a != b;
    }

  public:
    BlockDiscontinuity()
    {
        m_shared_data.first_element = new SmemType<Type4Byte>{BlockSize};
        m_shared_data.last_element  = new SmemType<Type4Byte>{BlockSize};
    };

    BlockDiscontinuity(tempStorage shared_data)
        : m_shared_data(shared_data) {};
    ~BlockDiscontinuity() = default;


    template <typename FlagOp>
    void FlagHeads(compute::ArrayVar<int, ITEMS_PER_THREAD>& head_flags,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp flag_op = Default_flag_op)
    {
        FlagHeads(head_flags, compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>{}, input, flag_op);
    };

    template <typename FlagOp>
    void FlagHeads(compute::ArrayVar<int, ITEMS_PER_THREAD>&       head_flags,
                   compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& prev_input,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp flag_op = Default_flag_op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        Int thid = Int(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        // $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            $if(i == 0)
            {
                $if(thid == 0)
                {
                    head_flags[i] = 1;
                }
                $else
                {
                    prev_input[i] = (*m_shared_data.last_element)[thid - 1];
                    head_flags[i] = ApplyOp(flag_op, input[i], prev_input[i], i);
                };
            }
            $else
            {
                prev_input[i] = input[i - 1];
                head_flags[i] = ApplyOp(flag_op, input[i], prev_input[i], i);
            };
        };
    };


    template <typename FlagOp>
    void FlagHeads(compute::ArrayVar<int, ITEMS_PER_THREAD>& head_flags,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                flag_op,
                   const Var<Type4Byte>& tile_predecessor_item)
    {
        FlagHeads(head_flags, compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>{}, input, flag_op, tile_predecessor_item);
    };

    template <typename FlagOp>
    void FlagHeads(compute::ArrayVar<int, ITEMS_PER_THREAD>&       head_flags,
                   compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& prev_input,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                flag_op,
                   const Var<Type4Byte>& tile_predecessor_item)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        Int thid = Int(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        // $for(i, 0u, compute::UInt(ITEMS_PER_THREAD))
        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            $if(i == 0)
            {
                $if(thid == 0)
                {
                    prev_input[i] = tile_predecessor_item;
                    head_flags[i] = ApplyOp(flag_op, input[i], prev_input[i], i);
                }
                $else
                {
                    prev_input[i] = (*m_shared_data.last_element)[thid - 1];
                    head_flags[i] = ApplyOp(flag_op, input[i], prev_input[i], i);
                };
            }
            $else
            {
                prev_input[i] = input[i - 1];
                head_flags[i] = ApplyOp(flag_op, input[i], prev_input[i], i);
            };
        };
    };


    template <typename FlagOp>
    void FlagTail(compute::ArrayVar<int, ITEMS_PER_THREAD>& tail_flags,
                  const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                  FlagOp flag_op = Default_flag_op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = UInt(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            $if(i == compute::UInt(ITEMS_PER_THREAD) - 1)
            {
                $if(thid == UInt(BlockSize - 1))
                {
                    tail_flags[i] = 1;
                }
                $else
                {
                    tail_flags[i] = ApplyOp(
                        flag_op, input[i], (*m_shared_data.first_element)[thid + 1], i);
                };
            }
            $else
            {
                tail_flags[i] = ApplyOp(flag_op, input[i], input[i + 1], i);
            };
        };
    };


    template <typename FlagOp>
    void FlagTail(compute::ArrayVar<int, ITEMS_PER_THREAD>& tail_flags,
                  const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                  FlagOp                                                flag_op,
                  const Var<Type4Byte>& tile_successor_item)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = UInt(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            $if(i == UInt(ITEMS_PER_THREAD) - 1)
            {
                $if(thid == UInt(BlockSize - 1))
                {
                    tail_flags[i] = ApplyOp(flag_op, input[i], tile_successor_item, i);
                }
                $else
                {
                    tail_flags[i] = ApplyOp(
                        flag_op, input[i], (*m_shared_data.first_element)[thid + 1], i);
                };
            }
            $else
            {
                tail_flags[i] = ApplyOp(flag_op, input[i], input[i + 1], i);
            };
        };
    };

    template <typename FlagOp>
    void FlagHeadsAndTails(compute::ArrayVar<int, ITEMS_PER_THREAD>& head_flags,
                           compute::ArrayVar<int, ITEMS_PER_THREAD>& tail_flags,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp flag_op = Default_flag_op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = UInt(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            $if(i == 0)
            {
                $if(thid == 0)
                {
                    head_flags[i] = 1;
                }
                $else
                {
                    head_flags[i] = ApplyOp(
                        flag_op, input[i], (*m_shared_data.last_element)[thid - 1], i);
                };
            }
            $else
            {
                head_flags[i] = ApplyOp(flag_op, input[i], input[i - 1], i);
            };

            $if(i == compute::UInt(ITEMS_PER_THREAD) - 1)
            {
                $if(thid == UInt(BlockSize - 1))
                {
                    tail_flags[i] = 1;
                }
                $else
                {
                    tail_flags[i] = ApplyOp(
                        flag_op, input[i], (*m_shared_data.first_element)[thid + 1], i);
                };
            }
            $else
            {
                tail_flags[i] = ApplyOp(flag_op, input[i], input[i + 1], i);
            };
        };
    };

    template <typename FlagOp>
    void FlagHeadsAndTails(compute::ArrayVar<int, ITEMS_PER_THREAD>& head_flags,
                           const Var<Type4Byte>& tile_predecessor_item,
                           compute::ArrayVar<int, ITEMS_PER_THREAD>& tail_flags,
                           const Var<Type4Byte>& tile_successor_item,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp flag_op)
    {
        using namespace luisa::compute;
        luisa::compute::set_block_size(BlockSize);
        UInt thid = UInt(thread_id().x);

        (*m_shared_data.first_element)[thid] = input[0];
        (*m_shared_data.last_element)[thid]  = input[ITEMS_PER_THREAD - 1];

        sync_block();

        for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            // head_flags
            $if(i == 0)
            {
                $if(thid == 0)
                {
                    head_flags[i] = ApplyOp(flag_op, input[i], tile_predecessor_item, i);
                }
                $else
                {
                    head_flags[i] =
                        ApplyOp(flag_op, input[i], m_shared_data.last_element[thid - 1], i);
                };
            }
            $else
            {
                head_flags[i] = ApplyOp(flag_op, input[i], input[i - 1], i);
            };

            // tail_flags
            $if(i == compute::UInt(ITEMS_PER_THREAD) - 1)
            {
                $if(thid == UInt(BlockSize - 1))
                {
                    tail_flags[i] = ApplyOp(flag_op, input[i], tile_successor_item, i);
                }
                $else
                {
                    tail_flags[i] =
                        ApplyOp(flag_op, input[i], m_shared_data.first_element[thid + 1], i);
                };
            }
            $else
            {
                tail_flags[i] = ApplyOp(flag_op, input[i], input[i + 1], i);
            };
        };
    };

  private:
    template <typename FlagOp>
    compute::Bool ApplyOp(FlagOp                         flag_op,
                          const compute::Var<Type4Byte>& a,
                          const compute::Var<Type4Byte>& b,
                          compute::UInt                  b_index)
    {
        if constexpr(std::is_invocable_v<FlagOp, Type4Byte, Type4Byte, int>)
        {
            return flag_op(a, b, b_index);
        }
        else
        {
            return flag_op(a, b);
        }
    }

    tempStorage m_shared_data;
};
}  // namespace luisa::parallel_primitive