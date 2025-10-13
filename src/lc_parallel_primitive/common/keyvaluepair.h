/*
 * @Author: Ligo 
 * @Date: 2025-09-26 15:47:22 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-26 16:11:54
 */


#pragma once
#include <lc_parallel_primitive/common/type_trait.h>
#include <luisa/dsl/struct.h>
#include <typeindex>

template <NumericT KeyType, NumericT ValueType>
struct KeyValuePair
{
    KeyType   key;
    ValueType value;
};

#define LUISA_KEY_VALUE_PAIR_TEMPLATE()                                        \
    template <NumericT KeyType, NumericT ValueType>
#define LUISA_KEY_VALUE_PAIR() KeyValuePair<KeyType, ValueType>

LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value){};


template <typename T>
struct is_key_value_pair : std::false_type
{
};

template <typename K, typename V>
struct is_key_value_pair<KeyValuePair<K, V>> : std::true_type
{
};

template <typename T>
concept KeyValuePairType = is_key_value_pair<T>::value;

// 值类型提取
template <typename T>
struct value_type_of
{
    // 对于非 KeyValuePair 类型，可能没有值类型
};

template <typename K, typename V>
struct value_type_of<KeyValuePair<K, V>>
{
    using type = V;
};

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;


template <typename KeyType, typename ValueType>
luisa::string get_key_value_shader_desc()
{
    luisa::string_view key_desc = luisa::compute::Type::of<KeyType>()->description();
    luisa::string_view value_desc = luisa::compute::Type::of<ValueType>()->description();

    return luisa::string(key_desc) + "+" + luisa::string(value_desc);
}

template <typename KeyType, typename ValueType, typename ReduceOp>
luisa::string get_key_value_op_shader_desc(ReduceOp op)
{
    luisa::string_view key_desc = luisa::compute::Type::of<KeyType>()->description();
    luisa::string_view value_desc = luisa::compute::Type::of<ValueType>()->description();

    return luisa::string(key_desc) + "+" + luisa::string(value_desc) + "+"
           + std::type_index(typeid(op)).name();
}