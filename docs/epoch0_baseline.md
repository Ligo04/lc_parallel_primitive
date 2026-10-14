# Epoch 0 — Baseline (pre-optimization)

Date: 2026-10-14
Hardware: NVIDIA GeForce RTX 5060 (8 GB, Blackwell sm_120, warp 32), Windows, MSVC 2026
CUDA: 13.4 (CUB/CCCL 3.4.0), driver runtime via LC cuda backend
LuisaCompute: bundled thirdparty at `LuisaCompute/` (xmake subproject, release mode)
lcpp: header-only library in `src/lcpp/`, commit base `3466506` + working-tree fixes (see below)

## Benchmark setup

- Binary: `build/windows/x64/release/lcpp_cub_bench.exe` (xmake target `lcpp_cub_bench`,
  companion CUDA baseline `lcpp_cub_bench_kernels` compiled by nvcc, gencode=native).
- Command: `lcpp_cub_bench <backend> [num_items] [repetitions]`; defaults 4,194,304 items, 10 reps.
- lcpp policies (compile-time macros in `tests/lcpp_cub_bench.cpp`):
  - Reduce: `DeviceReduce<256, 32, 4>`
  - Scan:   `DeviceScan<256, 32, 8>` (portable default; 320x25 would exceed the 32 KiB
    shared-memory floor of the DX/HLSL codegen path)
  - Radix:  `DeviceRadixSort<256, 32, 4>`
- Timing: CUDA backend uses CUevents on the LC stream (optional graph mode via
  `LCPP_BENCH_GRAPH`); dx/vk time the lcpp side with wall clock around
  enqueue + `stream.synchronize()` (host-timer overhead included, honest wall time);
  the CUB baseline always runs on the same physical GPU (own CUDA stream for dx/vk).
- Verification: every op is checked against a host reference and the CUB result;
  the bench exits non-zero on mismatch.

## Baseline numbers (median ms, lcpp vs CUB)

| op | backend | lcpp median | CUB median | lcpp/CUB |
|---|---|---|---|---|
| DeviceReduce.Sum (4M int) | cuda | 0.0325 | 0.0210 | 1.55x |
| DeviceScan.InclusiveSum | cuda | 0.1860 | 0.0803 | 2.32x |
| DeviceRadixSort.SortKeys | cuda | 0.7666 | 0.3418 | 2.24x |
| DeviceReduce.Sum | dx | 0.1741 | 0.0447 | 3.89x |
| DeviceScan.InclusiveSum | dx | 0.2937 | 0.0771 | 3.81x |
| DeviceRadixSort.SortKeys | dx | 0.9481 | 0.3432 | 2.76x |
| DeviceReduce.Sum | vk | 0.1481 | 0.0468 | 3.17x |
| DeviceScan.InclusiveSum | vk | 0.3246 | 0.0789 | 4.12x |
| DeviceRadixSort.SortKeys | vk | 3.8178 | 0.4235 | 9.02x |

Notes:
- dx/vk lcpp timings include per-op host sync overhead (~30-80 us floor); CUB is timed
  with pure GPU events on its own stream, so the ratio is pessimistic for small ops.
- vk radix sort is the worst outlier (9x) — worth a dedicated look (descriptor / ABI
  overheads on the XIR-SPIR-V path, policy fit).
- The reduce kernel launches 4096 partial blocks for 4M items and then recurses;
  scan/radix use decoupled-lookback one-sweep style agents.

## Working-tree fixes applied to reach this baseline (were required to build/run)

Build (LC thirdparty):
1. `LuisaCompute/src/backends/cuda/xmake.lua`: `lc-nvrtc` links `ntdll` on Windows
   (`nvrtc_static.lib` references `RtlGetLastNtStatus` on newer SDKs).

Bench harness (`tests/`):
2. New `tests/cub_bench_kernels.cu`: extern "C" wrappers around
   `cub::DeviceReduce::Sum` / `DeviceScan::InclusiveSum` / `DeviceRadixSort::SortKeys`
   (CUB 3.x API: temp bytes by `size_t&` reference).
3. `tests/lcpp_cub_bench.cpp`: rewritten to take `[backend] [num_items] [reps]`,
   supports dx/vk (host timer + standalone CUDA stream/buffers for the CUB baseline),
   real GPU name via `Context::backend_device_names` (skipped for vk: volk forbids
   enumeration while a device is live), policy macros adapted to the current lcpp API.
4. `tests/xmake.lua`: targets `lcpp_cub_bench_kernels` (nvcc, gencode native,
   `-Xcompiler=/Zc:preprocessor` for CCCL, `MD` runtime, devlink policy) and
   `lcpp_cub_bench`; CUDA env guarded by `CUDA_PATH` (targets disable themselves
   when absent).

Library (`src/lcpp`):
5. `device_radix_sort.h`: KEY_ONLY SortKeys passed **empty** dummy value views
   (`subview(0,0)`), which Vulkan rejects ("does not permit an empty storage-buffer
   descriptor"). Changed to 1-element dummy views (values are never touched when
   KEY_ONLY). This also fixed `device_radix_sort_one_sweep` on vk.

Configuration:
6. VK backend must be configured with `--lc_vk_backend_use_xir_spirv=true`
   (native XIR→SPIR-V path). The legacy HLSL→SPIR-V path cannot represent lcpp's
   descriptor-relative subview biases ("buffer-view offset is not descriptor-aligned").
   (This matches the machine default in LC's `scripts/options.lua`.)

Additional portability fixes (found by running the full test matrix):
7. Pair-typed temp buffers (KeyValuePair / IndexValuePair, `sizeof > alignof` for
   4-byte element types) were carved with `alignof` alignment, producing typed
   buffer views whose byte offset is not a multiple of the element stride --
   rejected by Vulkan. Fixed by aligning all such carvings to `sizeof(T)`
   (`device_reduce.h`, `device_scan.h`, `device_segment_reduce.h`). Fixes
   `device_scan_test` (scan-by-key) on vk.
8. `tests/decoupled_look_back.cpp` dispatched 102400 blocks in 1D, exceeding the
   D3D12 65535 group limit on dx. Reworked to 128 co-resident blocks that
   grid-stride over tiles (decoupled look-back requires co-residency; 400 blocks
   exceeded the concurrent-thread capacity and TDR-crashed dx/vk). All backends
   pass.

## Regression baseline

`docs/baseline_regression.log`: **all 21 test x backend combinations pass**
(7 tests x cuda/dx/vk). This is the green board every optimization epoch must
reproduce.
