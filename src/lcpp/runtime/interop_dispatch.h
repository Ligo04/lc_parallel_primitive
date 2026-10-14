/*
 * Epoch A of the vkCUDA interop campaign: the interop dispatch sink.
 *
 * `InteropContext` owns everything a lcpp device module needs in order to
 * launch a *cuda-backend-compiled* DSL kernel inside a Vulkan command buffer
 * through `VK_NV_cuda_kernel_launch` (LuisaCompute's `VkCudaInterop`
 * extension):
 *
 *   - the paired cuda backend `Device` (compile only: the produced module
 *     image targets the same physical GPU the Vulkan device is on),
 *   - the `VkCudaInterop` extension pointer of the *vk* device (import +
 *     launch),
 *   - the feature detection that decides whether the fast path is usable at
 *     all.
 *
 * Design contract (graceful degradation is the default):
 *
 *   InteropContext::create() returns `nullptr` -- after logging exactly one
 *   LUISA_WARNING -- whenever the fast path is not usable on this machine
 *   (extension missing, VK_NV_cuda_kernel_launch unsupported, no CUDA device
 *   LUID-matches the Vulkan device, cuda backend not installed, paired device
 *   creation failed).  A module that receives a null context keeps today's
 *   pure-Vulkan SPIR-V DSL dispatch path unchanged; interop is an opt-in
 *   accelerator, never an error.
 *
 * The imported kernel is an opaque module image: the vk side re-uses the ABI
 * documented in `luisa/backends/ext/vk_cuda_interop.h` (a single by-value
 * `struct Params` blob with 16-byte argument slots + an `ls_kid` uint4
 * trailer), so the call sites only swap the shader object they invoke.
 *
 * Lifetime (risk R6): `detail::InteropShader` holds `CudaShader` by value in
 * the per-module cache; a `CudaShader` destroys its vk-side function handle on
 * destruction, so every module holding interop shaders must be destroyed
 * before the Vulkan `Device` it dispatches on (the ordinary stack order in
 * lcpp: `Device device; ... DeviceReduce reduce;`).
 *
 * Per-op policy (epoch C): interop is not uniformly faster -- measured on
 * RTX 5060 / vk, the radix one-sweep wins 1.58x, reduce is at parity and scan
 * is ~3 % slower on the interop path (see .kimix_cache/epoch_C_report.md), so
 * `create()` consults the `LCPP_INTEROP` policy per module:
 *
 *   LCPP_INTEROP=1|all|on   every module takes the fast path (epoch A/B
 *                           behaviour, used by the vk-interop regression mode)
 *   LCPP_INTEROP=auto       lcpp's measured recommendation: reduce, segment
 *                           reduce and radix sort use it, scan stays on the
 *                           faster DSL path
 *   LCPP_INTEROP=a,b,c      explicit comma-separated module list
 *                           (reduce/scan/segment/radix, aliases accepted)
 *   LCPP_INTEROP=0|none|off (or unset) the pure-vulkan DSL path everywhere
 *
 * Both paths stay available for every op in every mode; the policy only picks
 * which one a module's `create()` binds to.
 */

#pragma once

#include <cstdlib>
#include <string_view>
#include <utility>

#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <lcpp/common/shader_cache.h>

namespace luisa::parallel_primitive
{

namespace detail
{
    /// One imported `VK_NV_cuda_kernel_launch` kernel, cached under the same
    /// `ShaderCacheKey` as the DSL shader it shadows.  Move-only: the wrapped
    /// `CudaShader` is the RAII owner of the imported vk-side function handle.
    struct InteropShader
    {
        vk_cuda_interop::CudaShader shader;

        InteropShader() noexcept = default;
        explicit InteropShader(vk_cuda_interop::CudaShader s) noexcept
            : shader{std::move(s)}
        {
        }
        InteropShader(InteropShader&&) noexcept            = default;
        InteropShader& operator=(InteropShader&&) noexcept = default;
        InteropShader(const InteropShader&)                = delete;
        InteropShader& operator=(const InteropShader&)     = delete;

        [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(shader); }
    };

    /// Cache of imported CUDA kernels (one per distinct kernel identity).
    using InteropShaderMap = luisa::unordered_map<ShaderCacheKey, InteropShader, ShaderCacheKeyHash>;

    /// The device modules that own an interop fast path.
    enum class InteropModule : unsigned
    {
        Reduce        = 1u << 0u,
        Scan          = 1u << 1u,
        SegmentReduce = 1u << 2u,
        RadixSort     = 1u << 3u,
    };

    /// lcpp's measured recommendation for `LCPP_INTEROP=auto`: the op where the
    /// cuda codegen wins (radix one-sweep), the ops at parity (reduce,
    /// segment reduce) -- but not scan, which the SPIR-V path implements ~3 %
    /// faster (epoch C measurement; see .kimix_cache/epoch_C_report.md).
    inline constexpr unsigned interop_recommended_mask =
        static_cast<unsigned>(InteropModule::Reduce) |
        static_cast<unsigned>(InteropModule::SegmentReduce) |
        static_cast<unsigned>(InteropModule::RadixSort);

    inline constexpr unsigned interop_all_mask =
        static_cast<unsigned>(InteropModule::Reduce) |
        static_cast<unsigned>(InteropModule::Scan) |
        static_cast<unsigned>(InteropModule::SegmentReduce) |
        static_cast<unsigned>(InteropModule::RadixSort);

    /// Parses one policy token; 0 means "not a module name".
    [[nodiscard]] inline unsigned interop_module_from_name(std::string_view name) noexcept
    {
        auto eq = [name](std::string_view candidate) noexcept
        { return name == candidate; };
        if(eq("reduce") || eq("reduction") || eq("device_reduce"))
        {
            return static_cast<unsigned>(InteropModule::Reduce);
        }
        if(eq("scan") || eq("device_scan") || eq("scanbykey") || eq("scan_by_key"))
        {
            return static_cast<unsigned>(InteropModule::Scan);
        }
        if(eq("segment") || eq("segmentreduce") || eq("segment_reduce") || eq("device_segment_reduce"))
        {
            return static_cast<unsigned>(InteropModule::SegmentReduce);
        }
        if(eq("radix") || eq("radixsort") || eq("radix_sort") || eq("sort") || eq("device_radix_sort"))
        {
            return static_cast<unsigned>(InteropModule::RadixSort);
        }
        return 0u;
    }

    /// Bitmask of the modules selected by the `LCPP_INTEROP` policy (parsed
    /// once per process).  Unknown tokens are ignored; a policy that names no
    /// known module disables the fast path.
    [[nodiscard]] inline unsigned interop_policy_mask() noexcept
    {
        static const auto mask = []() noexcept -> unsigned
        {
            const char* env = std::getenv("LCPP_INTEROP");
            if(env == nullptr || env[0] == '\0')
            {
                return 0u;
            }
            std::string_view policy{env};
            if(policy == "0" || policy == "none" || policy == "off" || policy == "false")
            {
                return 0u;
            }
            if(policy == "1" || policy == "all" || policy == "on" || policy == "true")
            {
                return interop_all_mask;
            }
            if(policy == "auto" || policy == "recommended")
            {
                return interop_recommended_mask;
            }
            auto selected = 0u;
            while(!policy.empty())
            {
                auto comma = policy.find(',');
                auto token = policy.substr(0u, comma);
                while(!token.empty() && (token.front() == ' ' || token.front() == '\t'))
                {
                    token.remove_prefix(1u);
                }
                while(!token.empty() && (token.back() == ' ' || token.back() == '\t'))
                {
                    token.remove_suffix(1u);
                }
                selected |= interop_module_from_name(token);
                if(comma == std::string_view::npos)
                {
                    break;
                }
                policy.remove_prefix(comma + 1u);
            }
            return selected;
        }();
        return mask;
    }
}  // namespace detail

class InteropContext;

/// True when the `LCPP_INTEROP` policy selects `module` (see the file comment
/// for the accepted values).  Accepts the same names as the policy tokens.
[[nodiscard]] inline bool interop_policy_selects(std::string_view module) noexcept
{
    auto bit = detail::interop_module_from_name(module);
    return bit != 0u && (detail::interop_policy_mask() & bit) != 0u;
}

/// Feature-detected vk-cuda interop sink.  One instance per process is enough;
/// lcpp modules borrow a pointer to it through their `create()`.
class InteropContext
{
  private:
    luisa::compute::Device         m_cuda_device;
    luisa::compute::VkCudaInterop* m_ext = nullptr;

  public:
    InteropContext() noexcept  = default;
    ~InteropContext() noexcept = default;

    /// Feature-detect the vk-cuda interop fast path for `device` (which must be
    /// the **vk** backend device) and, when it is usable, create the paired
    /// cuda backend device for kernel compilation.
    ///
    /// Returns `nullptr` when the fast path is unavailable; the caller must
    /// then keep the pure-Vulkan DSL path.  Exactly one LUISA_WARNING is
    /// emitted in that case (risk R3).
    [[nodiscard]] static luisa::unique_ptr<InteropContext> create(luisa::compute::Context& context,
                                                                  luisa::compute::Device& device) noexcept;

    /// True when the fast path is fully usable.
    [[nodiscard]] bool valid() const noexcept { return m_ext != nullptr; }

    /// True when the `LCPP_INTEROP` policy selects `module` (named exactly as
    /// the policy tokens: reduce / scan / segment / radix).
    [[nodiscard]] bool selects(std::string_view module) const noexcept
    {
        return valid() && interop_policy_selects(module);
    }

    /// The cuda backend device paired with the vk device (LUID-matched); used
    /// only to *compile* DSL kernels, never to dispatch them.
    [[nodiscard]] luisa::compute::Device& cuda_device() noexcept { return m_cuda_device; }

    [[nodiscard]] luisa::compute::VkCudaInterop* ext() noexcept { return m_ext; }

    /// Import a shader compiled by the cuda device into this Vulkan device.
    /// Fails closed: an empty `CudaShader` (evaluates to false) is returned
    /// when the shader cannot be imported (OptiX/printing/bound-argument
    /// kernels) -- callers fall back to the DSL path.
    template <size_t N, typename... Args>
    [[nodiscard]] vk_cuda_interop::CudaShader import(const luisa::compute::Shader<N, Args...>& shader) noexcept
    {
        if(m_ext == nullptr)
        {
            return {};
        }
        return m_ext->create_cuda_kernel(shader);
    }
};

inline luisa::unique_ptr<InteropContext> InteropContext::create(luisa::compute::Context& context,
                                                                luisa::compute::Device& device) noexcept
{
    auto unavailable = [](luisa::string_view reason) noexcept -> luisa::unique_ptr<InteropContext>
    {
        LUISA_WARNING(
            "lcpp: vk-cuda interop unavailable ({}); "
            "keeping the pure-vulkan DSL dispatch path.",
            reason);
        return nullptr;
    };

    auto ext = device.extension<luisa::compute::VkCudaInterop>();
    if(ext == nullptr)
    {
        return unavailable("VkCudaInterop extension is not present on this device");
    }
    if(!ext->cuda_kernel_launch_supported())
    {
        return unavailable("VK_NV_cuda_kernel_launch is not enabled");
    }
    if(ext->cuda_device_index() < 0)
    {
        return unavailable("no CUDA device LUID-matches this Vulkan device");
    }
    auto cuda_backend_installed = false;
    for(auto&& name : context.installed_backends())
    {
        if(name == "cuda")
        {
            cuda_backend_installed = true;
        }
    }
    if(!cuda_backend_installed)
    {
        return unavailable("the cuda backend is not installed");
    }

    luisa::compute::DeviceConfig cuda_config{.device_index = static_cast<size_t>(ext->cuda_device_index())};
    auto cuda_device = context.create_device("cuda", &cuda_config);
    if(!cuda_device)
    {
        return unavailable("could not create the paired cuda device");
    }

    auto interop           = luisa::make_unique<InteropContext>();
    interop->m_ext         = ext;
    interop->m_cuda_device = std::move(cuda_device);
    return interop;
}

/// Per-module interop selection used by every device module's `create()`:
/// returns `interop` when it is valid *and* the `LCPP_INTEROP` policy selects
/// `module`, and `nullptr` otherwise -- in which case the module keeps the
/// pure-vulkan DSL dispatch path.  This is the single place the epoch-C per-op
/// policy is applied, so call sites stay policy-agnostic.
[[nodiscard]] inline InteropContext* interop_for(InteropContext* interop, std::string_view module) noexcept
{
    if(interop == nullptr || !interop->valid() || !interop_policy_selects(module))
    {
        return nullptr;
    }
    return interop;
}

/// Cache-miss path of the interop fast path, shared by every device module:
/// import a freshly compiled cuda-backend shader into the vk device and store
/// it under `key` in `map`.
///
/// Fails **closed** (risks R1/R3): a null compilation result or an empty
/// `CudaShader` (OptiX / bound arguments / device printing) permanently
/// disables interop for the calling module -- `interop` is set to `nullptr`
/// and exactly one LUISA_WARNING is emitted -- so the caller falls back to the
/// pure-vulkan DSL dispatch path and never launches a broken module image.
///
/// Returns an iterator into `map`; `map.end()` means "no interop kernel, use
/// the DSL path" (which is also what the caller sees after a failure).
template <typename ShaderT>
[[nodiscard]] inline detail::InteropShaderMap::iterator
import_into_interop_cache(InteropContext*&                  interop,
                          detail::InteropShaderMap&         map,
                          const detail::ShaderCacheKey&     key,
                          const luisa::unique_ptr<ShaderT>& compiled_shader) noexcept
{
    if(interop == nullptr)
    {
        return map.find(key);
    }
    if(!compiled_shader)
    {
        LUISA_WARNING("lcpp: cuda-backend compilation of the interop kernel produced no "
                      "shader; falling back to the pure-vulkan DSL dispatch path.");
        interop = nullptr;
        return map.find(key);
    }
    auto imported = interop->import(*compiled_shader);
    if(!imported)
    {
        LUISA_WARNING("lcpp: vk-cuda interop import failed closed; "
                      "falling back to the pure-vulkan DSL dispatch path.");
        interop = nullptr;
        return map.find(key);
    }
    map.try_emplace(key, detail::InteropShader{std::move(imported)});
    return map.find(key);
}

}  // namespace luisa::parallel_primitive
