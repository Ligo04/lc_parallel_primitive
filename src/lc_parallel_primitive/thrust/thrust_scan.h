/*
 * @Author: Ligo 
 * @Date: 2025-09-22 10:58:05 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-22 11:01:02
 */
#pragma once

#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/type_trait.h>

namespace luisa::parallel_primitive
{
class ThrustScan : public LuisaModule
{
  public:
    ThrustScan()  = default;
    ~ThrustScan() = default;

    void create() {}

    template <NumericT Type4Byte>
    void InclusiveScan(CommandList&          cmdlist,
                       BufferView<Type4Byte> temp_buffer,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       int                   op = 0)
    {
        // select scan method
    }

    template <NumericT Type4Byte>
    void ExclusiveScan(CommandList&          cmdlist,
                       BufferView<Type4Byte> temp_buffer,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       int                   op = 0)
    {
        // select scan method
    }
};
}  // namespace luisa::parallel_primitive