/*
 * @Author: Ligo 
 * @Date: 2025-09-18 19:10:03 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-26 15:54:22
 */
#pragma once

#include <type_traits>

template <typename T>
static constexpr bool is_numeric_v =
    std::is_integral_v<T> || std::is_floating_point_v<T>;
template <typename T>
concept NumericT = is_numeric_v<T>;
