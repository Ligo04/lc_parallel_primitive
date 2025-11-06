/*
 * @Author: Ligo 
 * @Date: 2025-09-26 15:47:22 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 17:15:36
 */


#pragma once
#include "luisa/core/basic_traits.h"
#include <lcpp/common/type_trait.h>
#include <luisa/dsl/struct.h>
#include <typeindex>

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

// 值类型提取
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
}  // namespace luisa::parallel_primitive

#define LUISA_KEY_VALUE_PAIR_TEMPLATE() template <NumericT KeyType, NumericT ValueType>
#define LUISA_KEY_VALUE_PAIR() luisa::parallel_primitive::KeyValuePair<KeyType, ValueType>

LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value){};
