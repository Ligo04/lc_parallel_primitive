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
