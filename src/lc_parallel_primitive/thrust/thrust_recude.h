/*
 * @Author: Ligo 
 * @Date: 2025-09-22 10:51:36 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-22 11:26:29
 */

#pragma once

#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/type_trait.h>

namespace luisa::parallel_primitive
{
class ThrustReduce : public LuisaModule
{
    // Implementation details for ThrustReduce
  public:
    ThrustReduce()  = default;
    ~ThrustReduce() = default;

    void create() {}

    template <NumericT Type4Byte>
    void Reduce(CommandList&          cmdlist,
                BufferView<Type4Byte> temp_buffer,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                int                   op = 0)
    {
        // select reduce method
    }

  private:
    template <NumericT Type4Byte>
    void Reduce_Seq(CommandList&          cmdlist,
                    BufferView<Type4Byte> temp_buffer,
                    BufferView<Type4Byte> d_in,
                    BufferView<Type4Byte> d_out,
                    size_t                num_item,
                    int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Reduce_Binary(CommandList&          cmdlist,
                       BufferView<Type4Byte> temp_buffer,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Reduce_Trenary(CommandList&          cmdlist,
                        BufferView<Type4Byte> temp_buffer,
                        BufferView<Type4Byte> d_in,
                        BufferView<Type4Byte> d_out,
                        size_t                num_item,
                        int                   op = 0)
    {
    }
};
}  // namespace luisa::parallel_primitive