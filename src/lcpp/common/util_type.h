/*
 * @Author: Ligo 
 * @Date: 2025-09-26 15:47:22 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 16:46:18
 */


#pragma once
#include <luisa/dsl/struct.h>
#include <luisa/core/basic_traits.h>
#include <lcpp/common/type_trait.h>

namespace luisa::parallel_primitive
{
template <NumericT KeyType, NumericT ValueType>
struct KeyValuePair
{
    KeyType   key;
    ValueType value;
};


template <typename T>
struct is_key_value_pair : std::false_type
{
};

template <typename K, typename V>
struct is_key_value_pair<luisa::parallel_primitive::KeyValuePair<K, V>> : std::true_type
{
};

template <typename T>
concept KeyValuePairType = is_key_value_pair<T>::value;

template <typename T>
struct value_type_of
{
};

template <typename K, typename V>
struct value_type_of<luisa::parallel_primitive::KeyValuePair<K, V>>
{
    using type = V;
};

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;
template <typename T>
concept NumericTOrKeyValuePairT = NumericT<T> || KeyValuePairType<T>;
template <NumericT Type4Byte>
using IndexValuePairT = KeyValuePair<luisa::uint, Type4Byte>;


// Double Buffer(device) for Ping-Pong Buffering
template <typename T>
struct DoubleBuffer
{
    compute::BufferVar<T> d_buffer[2];
    compute::Int          selector;

    DoubleBuffer() = default;

    DoubleBuffer(compute::BufferVar<T> d_current, compute::BufferVar<T> d_alternate) noexcept
        : d_buffer{d_current, d_alternate}
        , selector{0}
    {
    }

    [[nodiscard]] compute::BufferVar<T> current() noexcept { return d_buffer[selector]; }

    [[nodiscard]] compute::BufferVar<T> alternate() noexcept { return d_buffer[selector ^ 1]; };
};


enum class Category : uint
{
    NOT_A_NUMBER      = 0,
    SIGNED_INTERGER   = 1,
    UNSIGNED_INTERGER = 2,
    FLOATING_POINT    = 3
};
namespace details
{
    struct is_primite_impl;

    template <Category _CATEGORY, bool _PRIMIRIVE, typename _UnsignedBits, typename T>
    struct BaseTraits
    {
      private:
        friend struct is_primite_impl;

        static constexpr bool is_primitive = _PRIMIRIVE;
    };

    template <typename _UnsignedBits, typename T>
    struct BaseTraits<Category::UNSIGNED_INTERGER, false, _UnsignedBits, T>
    {
      private:
        friend struct is_primite_impl;

        static constexpr bool is_primitive = false;
    };


}  // namespace details
}  // namespace luisa::parallel_primitive

#define LUISA_KEY_VALUE_PAIR_TEMPLATE() template <NumericT KeyType, NumericT ValueType>
#define LUISA_KEY_VALUE_PAIR() luisa::parallel_primitive::KeyValuePair<KeyType, ValueType>
LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value){};
