/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-19 17:27:12
 */

#pragma once

#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/type_trait.h>

namespace luisa::parallel_primitive
{
namespace detail
{

}
class DeviceReduce : public LuisaModule
{
    // Implementation details for DeviceReduce
  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

    void create() {}

    template <NumericT Type4Byte>
    void Reduce(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             BufferView<Type4Byte> temp_buffer,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item,
             int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
    }


    template <NumericT Type4Byte>
    void ReduceByKey(CommandList&          cmdlist,
                     BufferView<Type4Byte> temp_buffer,
                     BufferView<Type4Byte> d_keys_in,
                     BufferView<Type4Byte> d_values_in,
                     BufferView<Type4Byte> d_keys_out,
                     BufferView<Type4Byte> d_values_out,
                     size_t                num_item,
                     int                   op = 0)
    {
    }
};
}  // namespace luisa::parallel_primitive