#pragma once

#include <lc_parallel_primitive/type_trait.h>
#include <lc_parallel_primitive/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace detail
{

}
class Warpscan : public LuisaModule
{
    // warp level Scan
  public:
    Warpscan()  = default;
    ~Warpscan() = default;

    template <NumericT Type4Byte>
    void InclusiveSum(CommandList&          cmdlist,
                      BufferView<Type4Byte> temp_buffer,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_item,
                      int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void ExclusiveSum(CommandList&          cmdlist,
                      BufferView<Type4Byte> temp_buffer,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_item,
                      int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void InclusiveScan(CommandList&          cmdlist,
                       BufferView<Type4Byte> temp_buffer,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       int                   op = 0)
    {
    }


    template <NumericT Type4Byte>
    void ExclusiveScan(CommandList&          cmdlist,
                       BufferView<Type4Byte> temp_buffer,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       int                   op = 0)
    {
    }

    template <NumericT Type4Byte>
    void Scan(CommandList&          cmdlist,
              BufferView<Type4Byte> temp_buffer,
              BufferView<Type4Byte> d_in,
              BufferView<Type4Byte> d_out,
              size_t                num_item,
              int                   op = 0)
    {
    }
};
}  // namespace luisa::parallel_primitive