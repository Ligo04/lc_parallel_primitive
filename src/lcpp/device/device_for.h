/*
 * @Author: Ligo 
 * @Date: 2025-09-19 23:06:17 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 11:39:10
 */
#pragma once

#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;
class DeviceFor : public LuisaModule
{

    // using namespace luisa::compute;

  public:
    int  m_block_size    = 256;
    int  m_num_banks     = 32;
    int  m_log_mem_banks = 5;
    bool m_created       = false;


  public:
    DeviceFor()  = default;
    ~DeviceFor() = default;


    void create(Device& device)
    {
        int num_elements_per_block = m_block_size * 2;
        int extra_space            = num_elements_per_block / m_num_banks;
        m_created                  = true;
    }


    template <NumericT Type4Byte>
    void for_each(CommandList&          cmdlist,
                  BufferView<Type4Byte> temp_buffer,
                  size_t                num_item,
                  Callable<Type4Byte>   op)
    {
    }

    template <NumericT Type4Byte>
    void for_each_n(CommandList&          cmdlist,
                    BufferView<Type4Byte> temp_buffer,
                    size_t                num_item,
                    Callable<Type4Byte>   op)
    {
    }


    template <NumericT Type4Byte>
    void for_each_extents(CommandList&          cmdlist,
                          BufferView<Type4Byte> temp_buffer,
                          size_t                num_item,
                          Callable<Type4Byte>   op)
    {
    }

  private:
    template <NumericT Type4Byte>
    void compile(Device& device)
    {
        const auto n_blocks = m_block_size;
    }
};
}  // namespace luisa::parallel_primitive