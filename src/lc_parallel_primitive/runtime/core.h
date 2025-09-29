#pragma once
/**
 * @file core/runtime.h
 * @brief Some Encryptance for Luisa Runtime
 * @author sailing-innocent
 * @date 2023-06-25
 */

#include <luisa/luisa-compute.h>

namespace luisa::parallel_primitive
{

template <typename T>
using U = luisa::unique_ptr<T>;

template <typename T>
using S = luisa::shared_ptr<T>;

template <typename F>
using UCallable = U<luisa::compute::Callable<F>>;

template <size_t I, typename F, typename... Args>
inline void lazy_compile(luisa::compute::Device&                device,
                         U<luisa::compute::Shader<I, Args...>>& ushader,
                         F&&                                    func,
                         const luisa::compute::ShaderOption&    option = {
#ifndef NDEBUG
                             .enable_debug_info = true
#endif
                         }) noexcept
{
    using S = luisa::compute::Shader<I, Args...>;
    if(!ushader)
    {
        ushader = luisa::make_unique<S>(device.compile<I>(std::forward<F>(func), option));
    }
}


static inline float to_radius(float degree)
{
    return degree * 0.0174532925f;
}
static inline int imax(int a, int b)
{
    return a > b ? a : b;
}
static inline bool is_power_of_two(int x)
{
    return (x & (x - 1)) == 0;
}
static inline float radians(float degree)
{
    return degree * 0.017453292519943295769236907684886f;
}
static inline int floor_pow_2(int n)
{
#ifdef WIN32
    return 1 << (int)logb((float)n);
#else
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

static void get_temp_size_scan(size_t& temp_storage_size, size_t m_block_size, size_t num_items)
{
    auto         block_size       = m_block_size;
    unsigned int max_num_elements = num_items;
    temp_storage_size             = 0;
    unsigned int num_elements     = max_num_elements;  // input segment size
    int          level            = 0;
    do
    {
        // output segment size
        unsigned int num_blocks =
            imax(1, (int)ceil((float)num_elements / (2.f * block_size)));
        if(num_blocks > 1)
        {
            level++;
            temp_storage_size += num_blocks;
        }
        num_elements = num_blocks;
    } while(num_elements > 1);
    temp_storage_size += 1;
}

class LuisaModule : public vstd::IOperatorNewBase
{
  protected:
    using Context = luisa::compute::Context;
    template <typename T>
    using Var = luisa::compute::Var<T>;
    template <typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template <typename T>
    using BufferView = luisa::compute::BufferView<T>;
    template <typename T>
    using SmemType = luisa::compute::Shared<T>;
    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;
    template <typename T>
    using Image = luisa::compute::Image<T>;
    template <typename T>
    using ImageView = luisa::compute::ImageView<T>;
    template <size_t I, typename... Ts>
    using Shader = luisa::compute::Shader<I, Ts...>;
    template <size_t I, typename... Ts>
    using Kernel = luisa::compute::Kernel<I, Ts...>;


    using Device      = luisa::compute::Device;
    using CommandList = luisa::compute::CommandList;
    using float2      = luisa::float2;
    using float3      = luisa::float3;
    using float4      = luisa::float4;
    using float3x3    = luisa::float3x3;
    using float4x4    = luisa::float4x4;
    using uint        = luisa::uint;
    using uint2       = luisa::uint2;
    using uint3       = luisa::uint3;
    using ulong       = luisa::ulong;
    using double2     = luisa::double2;
    using double3     = luisa::double3;
    using double4     = luisa::double4;
    // using double3x3   = luisa::double3x3;
    // using double4x4   = luisa::double4x4;

    using Stream = luisa::compute::Stream;
    using Type   = luisa::compute::Type;

    int                        m_log_mem_banks = 5;
    inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
    {
        return i >> m_log_mem_banks;
    }
};


}  // namespace luisa::parallel_primitive