/*
 * @Author: Ligo 
 * @Date: 2025-09-28 15:37:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-29 11:44:03
 */
#pragma once

#include <lc_parallel_primitive/common/type_trait.h>
#include <luisa/dsl/func.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <luisa/dsl/builtin.h>


namespace luisa::parallel_primitive
{

enum class DefaultBlockScanAlgorithm
{
    NAIVE,
    SHARED_MEMORY,
    WARP_SHUFFLE
};
template <NumericT Type4Byte, size_t BlockSize = 256, DefaultBlockScanAlgorithm Algorithm = DefaultBlockScanAlgorithm::SHARED_MEMORY>
class BlockScan : public LuisaModule
{

    using ScanOpCallable = luisa::compute::Callable<void()>;

  public:
    BlockScan(SmemType<Type4Byte> temp_buffer)
        : m_shared_mem(temp_buffer)
    {
    }
    ~BlockScan() = default;

  public:
    template <typename ScanOp>
    ScanOpCallable ExclusiveScan(BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, ScanOp op)
    {
        return [&]()
        {

        };
    }
    template <typename ScanOp>
    ScanOpCallable InclusiveScan(BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, ScanOp op)
    {
        return [&]() {};
    }


    ScanOpCallable ExclusiveSum(BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out)
    {
        return [&]()
        {
            using namespace luisa::compute;

            luisa::compute::set_block_size(BlockSize);

            $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
            {
                Int block_size_ = Int(block_size());
                m_shared_mem    = SmemType<Type4Byte>(BlockSize);
                Int thid        = Int(thread_id().x);
                Int global_id   = Int(block_id().x * block_size_ + thid);

                m_shared_mem[thid] = d_in.read(global_id);
                sync_block();

                // up-sweep
                Int offset = def(1);
                $for(offset, 1, block_size_, offset * 2)
                {
                    UInt index = (thid + 1) * offset * 2 - 1;
                    $if(index < block_size_)
                    {
                        m_shared_mem[index] += m_shared_mem[index - offset];
                    };
                    sync_block();
                };

                $if(thid == 0)
                {
                    // clear the last element for exclusive scan
                    m_shared_mem[block_size_ - 1] = 0;
                };
                sync_block();

                // down-sweep
                $for(offset, block_size_ >> 1, 0, offset / 2)
                {
                    UInt index = (thid + 1) * offset * 2 - 1;
                    $if(index < block_size_)
                    {
                        m_shared_mem[index] += m_shared_mem[index - offset];
                    };
                    sync_block();
                };

                d_out.write(global_id, m_shared_mem[thid]);
            }
            $else{
                // LUISA_ERROR("Not Implemented Yet!");
            };
        };
    }
    ScanOpCallable InclusiveSum(BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out)
    {
        return [&]()
        {
            using namespace luisa::compute;

            luisa::compute::set_block_size(BlockSize);

            $if(Algorithm == DefaultBlockScanAlgorithm::SHARED_MEMORY)
            {
                Int block_size_ = Int(block_size());
                m_shared_mem    = SmemType<Type4Byte>(BlockSize);
                Int thid        = Int(thread_id().x);
                Int global_id   = Int(block_id().x * block_size_ + thid);

                m_shared_mem[thid] = d_in.read(global_id);
                sync_block();

                // up-sweep
                Int offset = def(1);
                $for(offset, 1, block_size_, offset * 2)
                {
                    UInt index = (thid + 1) * offset * 2 - 1;
                    $if(index < block_size_)
                    {
                        m_shared_mem[index] += m_shared_mem[index - offset];
                    };
                    sync_block();
                };

                $if(thid == 0)
                {
                    // clear the last element for exclusive scan
                    m_shared_mem[block_size_ - 1] = 0;
                };
                sync_block();

                // down-sweep
                $for(offset, block_size_ >> 1, 0, offset / 2)
                {
                    UInt index = (thid + 1) * offset * 2 - 1;
                    $if(index < block_size_)
                    {
                        m_shared_mem[index] += m_shared_mem[index - offset];
                    };
                    sync_block();
                };

                d_out.write(global_id, m_shared_mem[thid] + d_in.read(global_id));
            }
            $else{};
        };
    }

  private:
    SmemType<Type4Byte> m_shared_mem;
};
}  // namespace luisa::parallel_primitive