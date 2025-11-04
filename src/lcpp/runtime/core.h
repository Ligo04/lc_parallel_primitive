#pragma once
/**
 * @file core/runtime.h
 * @brief Some Encryptance for Luisa Runtime
 * @author sailing-innocent
 * @date 2023-06-25
 */

#include <cstddef>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/keyvaluepair.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{
namespace details
{
    constexpr static size_t WARP_SIZE        = 32;
    constexpr static size_t BLOCK_SIZE       = 256;
    constexpr static size_t ITEMS_PER_THREAD = 2;
};  // namespace details


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
};
}  // namespace luisa::parallel_primitive