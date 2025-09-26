/*
 * @Author: Ligo 
 * @Date: 2025-09-19 16:05:47 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-23 00:04:35
 */
#pragma once
//#include "template_struct.h"
#include <lc_parallel_primitive/warp/warp_scan.h>
#include <lc_parallel_primitive/device/device_reduce.h>


template <typename valueType, size_t Rows, size_t Cols>
struct MyMatrix
{
    std::array<valueType, Rows * Cols> data{};  // 用 array 存储数据
};

#define LUISA_MATERIX_TEMPLATE()                                               \
    template <typename valueType, size_t Rows, size_t Cols>
#define LUISA_MATERIX() MyMatrix<valueType, Rows, Cols>

LUISA_TEMPLATE_STRUCT(LUISA_MATERIX_TEMPLATE, LUISA_MATERIX, data){};
