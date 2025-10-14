/*
 * @Author: Ligo 
 * @Date: 2025-10-09 09:52:40 
 * @Last Modified by:   Ligo 
 * @Last Modified time: 2025-10-09 09:52:40 
 */


#pragma once
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lc_parallel_primitive/runtime/core.h>
#include <lc_parallel_primitive/common/type_trait.h>
#include <lc_parallel_primitive/common/keyvaluepair.h>
#include <lc_parallel_primitive/block/block_reduce.h>
#include <lc_parallel_primitive/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = 256, size_t ITEMS_PER_THREAD = 1>
class DeviceScan : public LuisaModule
{
  public:
    int m_block_size = BLOCK_SIZE;
    int m_warp_nums  = 32;

  private:
    int    m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

  public:
    DeviceScan()  = default;
    ~DeviceScan() = default;

    void Create(Device& device)
    {
        m_device                   = device;
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
    }

    template <NumericT Type4Byte>
    void ExclusiveSum(CommandList&          cmdlist,
                      Stream&               stream,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_item)
    {
    }

    template <NumericT Type4Byte>
    void InclusiveSum(CommandList&          cmdlist,
                      Stream&               stream,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_item)
    {
    }

    template <NumericT Type4Byte,
              typename Scanop = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void ExclusiveScan(CommandList&          cmdlist,
                       Stream&               stream,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       Scanop                scan_op)
    {
    }

    template <NumericT Type4Byte,
              typename Scanop = luisa::compute::Callable<Var<Type4Byte>(const Var<Type4Byte>&, Var<Type4Byte>&)>>
    void InclusiveScan(CommandList&          cmdlist,
                       Stream&               stream,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_item,
                       Scanop                scan_op)
    {
    }

    template <NumericT Type4Byte>
    void ExclusivSumByKey(CommandList&          cmdlist,
                          Stream&               stream,
                          BufferView<Type4Byte> d_keys_in,
                          BufferView<Type4Byte> d_values_in,
                          BufferView<Type4Byte> d_values_out,
                          size_t                num_item)
    {
    }

    template <NumericT Type4Byte>
    void InclusiveSumByKey(CommandList&          cmdlist,
                           Stream&               stream,
                           BufferView<Type4Byte> d_keys_in,
                           BufferView<Type4Byte> d_values_in,
                           BufferView<Type4Byte> d_values_out,
                           size_t                num_item)
    {
    }


  private:
};
}  // namespace luisa::parallel_primitive