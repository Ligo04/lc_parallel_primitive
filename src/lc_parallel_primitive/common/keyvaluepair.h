/*
 * @Author: Ligo 
 * @Date: 2025-09-26 15:47:22 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-26 16:11:54
 */


#pragma once
#include <lc_parallel_primitive/common/type_trait.h>
#include <luisa/dsl/struct.h>

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

template <typename KeyType, typename ValueType>
luisa::string get_key_value_shader_desc()
{
    luisa::string_view key_desc = luisa::compute::Type::of<KeyType>()->description();
    luisa::string_view value_desc = luisa::compute::Type::of<ValueType>()->description();

    return luisa::string(key_desc) + "+" + luisa::string(value_desc);
}