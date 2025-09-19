/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:19:18 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-19 16:02:27
 */
#pragma once
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/type_trait.h>

namespace luisa::parallel_primitive
{
class WarpExchange : public LuisaModule
{
    // warp level Exchange
  public:
    WarpExchange()  = default;
    ~WarpExchange() = default;

    template <NumericT Type4Byte>
    void Exchange(CommandList&          cmdlist,
                  BufferView<Type4Byte> d_in,
                  BufferView<Type4Byte> d_out,
                  size_t                num_item,
                  int                   op = 0)
    {
    }
};
}  // namespace luisa::parallel_primitive