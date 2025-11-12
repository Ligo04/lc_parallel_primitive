/*
 * @Author: Ligo 
 * @Date: 2025-09-26 15:47:22 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-13 00:20:19
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
    compute::BufferView<T> d_buffer[2];
    int                    selector;

    DoubleBuffer() = default;

    DoubleBuffer(compute::BufferView<T> d_current, compute::BufferView<T> d_alternate) noexcept
        : d_buffer{d_current, d_alternate}
        , selector{0}
    {
    }

    [[nodiscard]] compute::BufferView<T> current() noexcept { return d_buffer[selector]; }

    [[nodiscard]] compute::BufferView<T> alternate() noexcept { return d_buffer[selector ^ 1]; };
};


enum class Category : uint
{
    NOT_A_NUMBER     = 0u,
    SIGNED_INTEGER   = 1u,
    UNSIGNED_INTEGER = 2u,
    FLOATING_POINT   = 3u
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
    struct BaseTraits<Category::UNSIGNED_INTEGER, true, _UnsignedBits, T>
    {
        using UnsignedBits                       = _UnsignedBits;
        static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
        static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

        static compute::Var<UnsignedBits> TwiddleIn(const compute::Var<UnsignedBits>& key)
        {
            return key;
        }
        static compute::Var<UnsignedBits> TwiddleOut(const compute::Var<UnsignedBits>& key)
        {
            return key;
        }

      private:
        friend struct is_primite_impl;
        static constexpr bool is_primitive = true;
    };

    template <typename _UnsignedBits, typename T>
    struct BaseTraits<Category::SIGNED_INTEGER, true, _UnsignedBits, T>
    {
        using UnsignedBits = _UnsignedBits;
        static constexpr UnsignedBits HIGH_BIT = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
        static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
        static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

        static compute::Var<UnsignedBits> TwiddleIn(const compute::Var<UnsignedBits>& key)
        {
            return key ^ compute::Var<UnsignedBits>(HIGH_BIT);
        }
        static compute::Var<UnsignedBits> TwiddleOut(const compute::Var<UnsignedBits>& key)
        {
            return key ^ compute::Var<UnsignedBits>(HIGH_BIT);
        }

      private:
        friend struct is_primite_impl;
        static constexpr bool is_primitive = true;
    };

    template <typename _UnsignedBits, typename T>
    struct BaseTraits<Category::FLOATING_POINT, true, _UnsignedBits, T>
    {
        using UnsignedBits = _UnsignedBits;
        static constexpr UnsignedBits HIGH_BIT = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
        static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(-1);
        static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

        static compute::Var<UnsignedBits> TwiddleIn(const compute::Var<UnsignedBits>& key)
        {
            auto mask = (key & compute::Var<UnsignedBits>(HIGH_BIT)) ?
                            compute::Var<UnsignedBits>(-1) :
                            compute::Var<UnsignedBits>(HIGH_BIT);
            return key ^ mask;
        }
        static compute::Var<UnsignedBits> TwiddleOut(const compute::Var<UnsignedBits>& key)
        {
            auto mask = (key & compute::Var<UnsignedBits>(HIGH_BIT)) ?
                            compute::Var<UnsignedBits>(-1) :
                            compute::Var<UnsignedBits>(HIGH_BIT);
            return key ^ mask;
        }

      private:
        friend struct is_primite_impl;
        static constexpr bool is_primitive = true;
    };
}  // namespace details

template <Category _CATEGORY, bool _PRIMITIVE, typename _UnsignedBits, typename T>
using BaseTraits = details::BaseTraits<_CATEGORY, _PRIMITIVE, _UnsignedBits, T>;

template <typename T>
struct NumericTraits : BaseTraits<Category::NOT_A_NUMBER, false, T, T>
{
};
template <>
struct NumericTraits<char>
    : BaseTraits<(std::numeric_limits<char>::is_signed) ? Category::SIGNED_INTEGER : Category::UNSIGNED_INTEGER, true, unsigned char, char>
{
};
template <>
struct NumericTraits<signed char> : BaseTraits<Category::SIGNED_INTEGER, true, uchar, signed char>
{
};
template <>
struct NumericTraits<short> : BaseTraits<Category::SIGNED_INTEGER, true, unsigned short, short>
{
};
template <>
struct NumericTraits<int> : BaseTraits<Category::SIGNED_INTEGER, true, uint, int>
{
};
template <>
struct NumericTraits<long> : BaseTraits<Category::SIGNED_INTEGER, true, ulong, long>
{
};
template <>
struct NumericTraits<slong> : BaseTraits<Category::SIGNED_INTEGER, true, ulong, slong>
{
};
template <>
struct NumericTraits<uchar> : BaseTraits<Category::UNSIGNED_INTEGER, true, uchar, uchar>
{
};

template <>
struct NumericTraits<uint> : BaseTraits<Category::UNSIGNED_INTEGER, true, uint, uint>
{
};
template <>
struct NumericTraits<ulong> : BaseTraits<Category::UNSIGNED_INTEGER, true, ulong, ulong>
{
};
// clang-format on

template <typename T>
using Traits = NumericTraits<std::remove_cv_t<T>>;

}  // namespace luisa::parallel_primitive

#define LUISA_KEY_VALUE_PAIR_TEMPLATE() template <NumericT KeyType, NumericT ValueType>
#define LUISA_KEY_VALUE_PAIR() luisa::parallel_primitive::KeyValuePair<KeyType, ValueType>
LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value){};
