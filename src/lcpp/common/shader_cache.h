/*
 * Epoch 1: cheap, allocation-free key for the per-module shader caches.
 *
 * The device dispatch APIs (DeviceReduce / DeviceScan / DeviceRadixSort /
 * DeviceSegmentReduce) cache one compiled shader per (data-type, operator)
 * combination in a per-module-instance map.  The original key was a
 * `luisa::string` rebuilt on EVERY call by concatenating
 * `Type::of<T>()->description()` with `typeid(op).name()`; building that string
 * (allocation + long-string hashing) dominated the per-call host overhead
 * (~100-140 us) for small inputs.
 *
 * A shader is fully determined by compile-time quantities only: the data
 * element type(s) and the operator type(s).  We therefore key on a handful of
 * stable pointer values -- one `const luisa::compute::Type*` per data type
 * (a process-wide singleton from the type registry) and one `const
 * std::type_info*` per operator (unique per operator type, including distinct
 * lambdas) -- instead of a string.  Key construction, hashing and comparison
 * are all O(1) with no heap traffic, and every distinct operator type still
 * maps to its own shader, so a single module instance called with different
 * ops resolves the right cached shader exactly as before.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <typeinfo>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/dsl/resource.h>

namespace luisa::parallel_primitive::detail
{

    /// A small fixed array of stable, process-wide identity pointers plus a
    /// component count.  Equality compares the (few) pointers; the hash mixes
    /// them with a splitmix-style avalanche.
    struct ShaderCacheKey {
        static constexpr int MAX_SLOTS = 6;
        std::array<const void*, MAX_SLOTS> ptrs{};
        uint8_t count = 0;

        [[nodiscard]] bool operator==(const ShaderCacheKey& rhs) const noexcept
        {
            if(count != rhs.count) {
                return false;
            }
            for(int i = 0; i < count; ++i) {
                if(ptrs[i] != rhs.ptrs[i]) {
                    return false;
                }
            }
            return true;
        }
    };

    struct ShaderCacheKeyHash {
        [[nodiscard]] size_t operator()(const ShaderCacheKey& k) const noexcept
        {
            // splitmix64 avalanche over the pointer values.
            auto mix = [](uint64_t x) noexcept {
                x += 0x9E3779B97F4A7C15ull;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                return x ^ (x >> 31);
            };
            uint64_t h = static_cast<uint64_t>(k.count);
            for(int i = 0; i < k.count; ++i) {
                h = mix(h ^ reinterpret_cast<uint64_t>(k.ptrs[i]));
            }
            return static_cast<size_t>(h);
        }
    };

    /// Build a key from up to ShaderCacheKey::MAX_SLOTS identity pointers.
    template <typename... Ts>
    [[nodiscard]] inline ShaderCacheKey make_shader_cache_key(const Ts*... ps) noexcept
    {
        ShaderCacheKey key;
        const void*    raw[] = {static_cast<const void*>(ps)...};
        key.count            = static_cast<uint8_t>(sizeof...(ps));
        for(uint8_t i = 0; i < key.count; ++i) {
            key.ptrs[i] = raw[i];
        }
        return key;
    }

    /// Identity pointer for a data element type (the process-wide type-registry
    /// singleton returned by Type::of<T>()).  Distinct descriptions give
    /// distinct pointers, so this discriminates exactly the same set of types the
    /// previous description-based string key did -- without building any string.
    template <typename T>
    [[nodiscard]] inline const void* type_ptr() noexcept
    {
        return static_cast<const void*>(luisa::compute::Type::of<T>());
    }

    /// Identity pointer for an operator *type*.  The operator object is only
    /// used to deduce its static type; typeid on an lvalue returns the static
    /// type, so this works for lambdas (unique closure type per definition site)
    /// and named functors alike -- matching the old typeid(op).name() semantics.
    template <typename Op>
    [[nodiscard]] inline const void* op_ptr(const Op&) noexcept
    {
        return static_cast<const void*>(std::addressof(typeid(Op)));
    }

    /// One of two distinct, stable addresses used to fold a boolean template
    /// parameter (e.g. KEY_ONLY / IS_DESCENDING) into a cache key.  Inline
    /// function-local statics have a single instance across TUs, so each value
    /// always maps to the same pointer.
    [[nodiscard]] inline const void* bool_ptr(bool b) noexcept
    {
        static const char k_false = 0;
        static const char k_true  = 1;
        return b ? static_cast<const void*>(&k_true) : static_cast<const void*>(&k_false);
    }

    /// Value type stored in every shader cache: a type-erased compiled shader
    /// Resource (reinterpreted back to its concrete Shader<K> at the call site).
    using ShaderCacheMap =
        luisa::unordered_map<ShaderCacheKey,
                             luisa::shared_ptr<luisa::compute::Resource>,
                             ShaderCacheKeyHash>;

}  // namespace luisa::parallel_primitive::detail
