# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

lcpp (LuisaCompute Parallel Primitives) 是一个基于 LuisaCompute 的 GPU 并行图元库，提供从线程级到设备级的高性能并行原语实现。这是一个 C++20 header-only 库。

## 构建命令

### XMake 构建 (推荐)
```bash
# 配置并构建
xmake f -m release
xmake

# 构建测试
xmake build -g test
```

### CMake 构建
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### 运行测试
```bash
# XMake
xmake run block_level_test
xmake run warp_level_test
xmake run device_reduce_test
xmake run device_scan_test
xmake run device_segment_reduce
xmake run device_radix_sort_one_sweep
xmake run decoupled_look_back

# CMake (在build目录)
ctest
./tests/block_level_test
```

## 代码架构

库采用分层架构,自下而上为:

```
Device级 (DeviceReduce, DeviceScan, DeviceRadixSort, DeviceSegmentReduce)
    ↓
Agent层 (算法策略管理,如 OneSweepSmallKeyTunedPolicy)
    ↓
Block级 (BlockReduce, BlockScan, BlockLoad, BlockStore, BlockExchange, BlockRadixRank)
    ↓
Warp级 (WarpReduce, WarpScan, WarpExchange) - 32线程粒度
    ↓
Thread级 (ThreadReduce, ThreadScan)
```

### 核心目录
- `src/lcpp/device/` - 设备级操作,调用 Agent 层实现大规模数据处理
- `src/lcpp/agent/` - 算法策略和调优参数
- `src/lcpp/block/` - 块级操作,典型配置 256 线程
- `src/lcpp/warp/` - Warp 级操作,使用 shuffle 或 shared memory 实现
- `src/lcpp/runtime/core.h` - LuisaModule 基类和 Shader 编译宏

### 关键设计模式
1. 所有并行操作类继承自 `LuisaModule`,提供统一的类型定义和 Shader 编译接口
2. 使用 `lazy_compile` 宏进行延迟 Shader 编译
3. 算法策略通过 Policy 类模板参数配置(如 `WarpScanAlgorithm::WARP_SHUFFLE`)
4. Block 操作支持多种算法:`SHARED_MEMORY` 和 `WARP_SHUFFLE`

## 依赖

- LuisaCompute - GPU 计算框架(自动通过 FetchContent 下载)
- boost_ut - 单元测试框架
- cpptrace - 堆栈跟踪

## 代码风格

遵循 `.clang-format` 配置:LLVM 风格,100 字符列宽,4 空格缩进。
