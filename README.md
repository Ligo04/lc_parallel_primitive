# LCPP - LuisaCompute Parallel Primitives

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)

A high-performance GPU parallel primitives library built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), providing thread-level to device-level parallel operations. LCPP is a header-only C++20 library inspired by CUDA CUB (Cooperative primitives for CUDA C++).

## Features

LCPP provides a comprehensive set of parallel primitives organized in a hierarchical architecture:

```
Device Level  → DeviceReduce, DeviceScan, DeviceRadixSort, DeviceSegmentReduce
       ↓
Agent Layer   → Algorithm policy management (e.g., OneSweepSmallKeyTunedPolicy)
       ↓
Block Level   → BlockReduce, BlockScan, BlockLoad, BlockStore, BlockRadixRank
       ↓
Warp Level    → WarpReduce, WarpScan, WarpExchange (32 threads)
       ↓
Thread Level  → ThreadReduce, ThreadScan
```

### Key Features

- **Header-only library** - Easy integration without compilation
- **Multi-backend support** - Leverages LuisaCompute's backend abstraction (CUDA, Metal, DirectX, Vulkan)
- **Lazy compilation** - Shaders are compiled on-demand using the `lazy_compile` macro
- **Flexible algorithms** - Multiple algorithm implementations (SHARED_MEMORY, WARP_SHUFFLE)
- **Type-safe** - Modern C++20 with strong type checking
- **Performance-focused** - Optimized policies and tuning parameters

## Build

### Prerequisites

- C++20 compatible compiler
- CMake 3.26+ or XMake 3.0+
- LuisaCompute (automatically fetched if not found)

### XMake (Recommended)

```bash
# Configure and build
xmake f -m release
xmake

# Build tests
xmake build -g test

# Run tests
xmake run block_level_test
xmake run warp_level_test
xmake run device_reduce_test
xmake run device_scan_test
xmake run device_segment_reduce
xmake run device_radix_sort_one_sweep
```

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run tests
ctest
# Or run individual tests
./tests/block_level_test
./tests/device_scan_test
```

## Quick Start

```cpp
#include <lcpp/parallel_primitive.h>
using namespace luisa::parallel_primitive;

// Create device and stream
Context ctx{argv[0]};
Device device = ctx.create_device("cuda");
Stream stream = device.create_stream();

// Device-level reduce example
DeviceReduce<> device_reduce;
device_reduce.create(device);

Buffer<int> input = device.create_buffer<int>(1024);
Buffer<int> output = device.create_buffer<int>(1);

CommandList cmdlist;
device_reduce.Sum(cmdlist, stream, input, output, 1024);
stream << cmdlist.commit() << synchronize();
```

## Implemented Features

### ✅ Thread Level
- [x] **ThreadReduce** - Thread-level reduction operations
- [x] **ThreadScan** - Thread-level scan (prefix sum) operations

### ✅ Warp Level (32 threads)
- [x] **WarpReduce** - Warp-level reduction (Sum, Min, Max, custom operators)
- [x] **WarpScan** - Warp-level inclusive/exclusive scan
- [x] **WarpExchange** - Data exchange patterns within a warp

### ✅ Block Level (typically 256 threads)
- [x] **BlockReduce** - Block-level reduction with SHARED_MEMORY and WARP_SHUFFLE algorithms
- [x] **BlockScan** - Block-level inclusive/exclusive scan with prefix callback support
- [x] **BlockLoad** - Efficient block-wide data loading (DIRECT, STRIPED modes)
- [x] **BlockStore** - Efficient block-wide data storing
- [x] **BlockExchange** - Data rearrangement within a block
- [x] **BlockRadixRank** - Ranking operations for radix sort
- [x] **BlockDiscontinuity** - Flag head/tail discontinuities in sequences

### ✅ Device Level
- [x] **DeviceReduce** - Device-wide reduction (Sum, Min, Max, custom operators)
- [x] **DeviceScan** - Device-wide inclusive/exclusive scan with decoupled look-back
- [x] **DeviceRadixSort** - Radix sort with OneSweep algorithm (SortKeys, SortPairs)
- [x] **DeviceSegmentReduce** - Segmented reduction operations
- [x] **DeviceHistogram** - Histogram computation
- [x] **DeviceFor** - Parallel for-loop utilities

## TODO List (Compared to CUDA CUB)

This section tracks missing features compared to the CUDA CUB library. Contributions are welcome!

### Priority 1: High-Priority Device Operations

#### DeviceSelect
- [ ] `DeviceSelect::Flagged` - Select items based on selection flags
- [ ] `DeviceSelect::If` - Select items based on predicate
- [ ] `DeviceSelect::Unique` - Select unique items from input sequence
- [ ] `DeviceSelect::UniqueByKey` - Select unique keys from key-value pairs

#### DevicePartition
- [ ] `DevicePartition::Flagged` - Partition based on selection flags
- [ ] `DevicePartition::If` - Partition based on predicate

#### DeviceReduce Extensions
- [ ] `DeviceReduce::ReduceByKey` - Reduce segments defined by keys
- [ ] `DeviceReduce::ArgMin` - Find minimum value and its index
- [ ] `DeviceReduce::ArgMax` - Find maximum value and its index

#### DeviceScan Extensions
- [ ] `DeviceScan::InclusiveScanByKey` - Segmented inclusive scan with keys
- [ ] `DeviceScan::ExclusiveScanByKey` - Segmented exclusive scan with keys
- [ ] Complete key-based scan implementation (partial implementation exists)

### Priority 2: Sorting and Merging

#### DeviceMergeSort
- [ ] `DeviceMergeSort::SortKeys` - Stable merge sort for keys
- [ ] `DeviceMergeSort::SortPairs` - Stable merge sort for key-value pairs
- [ ] `DeviceMergeSort::StableSortKeys` - Guaranteed stable sort
- [ ] `DeviceMergeSort::StableSortPairs` - Guaranteed stable sort for pairs

#### DeviceSegmentedSort
- [ ] `DeviceSegmentedRadixSort` - Radix sort within segments
- [ ] `DeviceSegmentedSort` - General sorting within segments

#### BlockMergeSort
- [ ] `BlockMergeSort::Sort` - Block-level stable merge sort
- [ ] `BlockMergeSort::SortBlockedToStriped` - Sort with layout transformation

### Priority 3: Advanced Operations

#### DeviceRunLengthEncode
- [ ] `DeviceRunLengthEncode::Encode` - Run-length encoding
- [ ] `DeviceRunLengthEncode::NonTrivialRuns` - Encode runs with length > 1

#### DeviceAdjacentDifference
- [ ] `DeviceAdjacentDifference::SubtractLeft` - In-place left subtraction
- [ ] `DeviceAdjacentDifference::SubtractLeftCopy` - Copy with left subtraction
- [ ] `DeviceAdjacentDifference::SubtractRight` - In-place right subtraction
- [ ] `DeviceAdjacentDifference::SubtractRightCopy` - Copy with right subtraction

#### DeviceSpmv (Sparse Matrix Operations)
- [ ] `DeviceSpmv::CsrMV` - Sparse matrix-vector multiplication (CSR format)

#### DeviceCopy
- [ ] `DeviceCopy::Batched` - Batched memory copy operations

### Priority 4: Block-Level Extensions

#### BlockRadixSort
- [ ] `BlockRadixSort::Sort` - Complete radix sort (not just ranking)
- [ ] `BlockRadixSort::SortDescending` - Descending radix sort
- [ ] `BlockRadixSort::SortBlockedToStriped` - Sort with layout change

#### BlockHistogram
- [ ] `BlockHistogram::Composite` - Block-level histogram computation
- [ ] `BlockHistogram::Init` - Initialize histogram bins

#### BlockAdjacentDifference
- [ ] `BlockAdjacentDifference::SubtractLeft` - Block-level left subtraction
- [ ] `BlockAdjacentDifference::SubtractRight` - Block-level right subtraction
- [ ] `BlockAdjacentDifference::FlagHeads` - Flag heads of segments
- [ ] `BlockAdjacentDifference::FlagTails` - Flag tails of segments

#### BlockRunLengthDecode
- [ ] `BlockRunLengthDecode::RunLengthDecode` - Decode run-length encoded data

#### BlockShuffle
- [ ] `BlockShuffle::Up` - Shuffle data up within block
- [ ] `BlockShuffle::Down` - Shuffle data down within block
- [ ] `BlockShuffle::Offset` - Shuffle data by offset

### Priority 5: Warp-Level Extensions

#### WarpMergeSort
- [ ] `WarpMergeSort::Sort` - Warp-level stable merge sort

#### WarpLoad/WarpStore
- [ ] `WarpLoad` - Optimized warp-level data loading
- [ ] `WarpStore` - Optimized warp-level data storing

#### WarpScan Extensions
- [ ] `WarpScan::Broadcast` - Broadcast value across warp

#### WarpReduce Extensions
- [ ] `WarpReduce::HeadSegmentedReduce` - Segmented reduce with head flags
- [ ] `WarpReduce::TailSegmentedReduce` - Segmented reduce with tail flags

### Priority 6: Thread-Level Extensions

#### ThreadLoad/ThreadStore
- [ ] `ThreadLoad` - Thread-level load with cache modifiers (CG, CA, CS, CV)
- [ ] `ThreadStore` - Thread-level store with cache modifiers (CG, CS, WT, WB)

### Algorithm Improvements
- [ ] Add SHARED_MEMORY implementations for Warp operations (currently only WARP_SHUFFLE)
- [ ] Add TRANSPOSE and WARP_TRANSPOSE modes for BlockLoad
- [ ] Add VECTORIZE mode for BlockLoad/BlockStore
- [ ] Optimize policies for different GPU architectures
- [ ] Add support for 64-bit indexing for large data

## Architecture

### Core Design Patterns

1. **LuisaModule Base Class** - All parallel operations inherit from `LuisaModule`, providing unified type definitions and shader compilation interface
2. **Lazy Shader Compilation** - Shaders are compiled on first use via `lazy_compile` macro
3. **Policy-Based Configuration** - Algorithm strategies configured through Policy template parameters
4. **Multiple Algorithm Support** - Block operations support both `SHARED_MEMORY` and `WARP_SHUFFLE` algorithms

### Directory Structure

```
src/lcpp/
├── device/           # Device-level operations
│   ├── details/      # Implementation details
│   └── device_*.h    # Public interfaces
├── agent/            # Algorithm policies and tuning
├── block/            # Block-level operations (256 threads)
│   └── detail/       # Implementation details
├── warp/             # Warp-level operations (32 threads)
│   └── details/      # Implementation details
├── thread/           # Thread-level operations
├── common/           # Common utilities and type traits
└── runtime/          # LuisaModule base and core definitions
```

### Key Files

- `src/lcpp/parallel_primitive.h` - Main header including all primitives
- `src/lcpp/runtime/core.h` - `LuisaModule` base class and `lazy_compile` macro
- `src/lcpp/agent/policy.h` - Tuning policies and parameters

## Code Style

- **Standard**: LLVM style with modifications (see `.clang-format`)
- **Line Width**: 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Language**: C++20

## Dependencies

- **LuisaCompute** - GPU compute framework (automatically fetched via CMake FetchContent)
- **boost_ut** - Unit testing framework
- **cpptrace** - Stack tracing utility

## Documentation

For detailed API documentation, see the header files in `src/lcpp/`. Each primitive class is documented with:
- Template parameters
- Member functions
- Algorithm options
- Usage examples

## Contributing

Contributions are welcome! Please feel free to:
1. Pick an item from the TODO list
2. Implement the feature following existing code patterns
3. Add tests in the `tests/` directory
4. Submit a pull request

### Implementation Guidelines

- Follow the existing class hierarchy (inherit from `LuisaModule`)
- Use `lazy_compile` for shader compilation
- Support multiple algorithm variants when applicable
- Add comprehensive tests
- Document public APIs

## References

- [LuisaCompute Documentation](https://github.com/LuisaGroup/LuisaCompute)
- [CUDA CUB Library](https://nvidia.github.io/cccl/cub/)
- [CUDA CUB Documentation](https://docs.nvidia.com/cuda/cub/index.html)

## License

[Specify your license here]

## Acknowledgments

- Built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) by LuisaGroup
- Inspired by NVIDIA's [CUB library](https://github.com/NVIDIA/cccl)

---

**Sources:**
- [CUB :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cub/index.html)
- [CUB — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cub/)
