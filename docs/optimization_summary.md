# lcpp Optimization Campaign — Final Summary

Date: 2026-10-14
Hardware: NVIDIA GeForce RTX 5060 (8 GB, Blackwell sm_120), Windows, MSVC 2026, CUDA 13.4 (CUB/CCCL 3.4)
Benchmark: `tests/lcpp_cub_bench.cpp` (xmake target `lcpp_cub_bench` + nvcc-compiled
`lcpp_cub_bench_kernels` CUB baseline), 4,194,304 items, median of 10 reps, results
verified **exactly** against a host reference and CUB on every run.
Measurement: cuda = CUDA events on the LC stream; dx/vk = amortized wall clock
(`HostSyncTimer`, batch=10; `LCPP_BENCH_BATCH=1` reproduces the epoch-0/1 mode).

## Results (median ms, lcpp vs CUB)

| op | backend | epoch 0 | final | speedup | lcpp/CUB: epoch 0 → final |
|---|---|---|---|---|---|
| DeviceReduce.Sum | cuda | 0.0325 | 0.0223 | 1.46x | 1.55x → **1.11x** |
| DeviceScan.InclusiveSum | cuda | 0.1860 | 0.0774 | 2.40x | 2.32x → **0.96x** |
| DeviceRadixSort.SortKeys | cuda | 0.7666 | 0.4285 | 1.79x | 2.24x → **1.00x** |
| DeviceReduce.Sum | dx | 0.1748¹ | 0.0434 | 4.0x¹ | 3.89x¹ → **0.93x** |
| DeviceScan.InclusiveSum | dx | 0.2937¹ | 0.1005 | 2.9x¹ | 3.81x¹ → **1.24x** |
| DeviceRadixSort.SortKeys | dx | 0.9481¹ | 0.6718 | 1.4x¹ | 2.76x¹ → **1.58x** |
| DeviceReduce.Sum | vk | 0.1481¹ | 0.0362 | 4.1x¹ | 3.17x¹ → **0.78x** |
| DeviceScan.InclusiveSum | vk | 0.3246¹ | 0.0921 | 3.5x¹ | 4.12x¹ → **1.15x** |
| DeviceRadixSort.SortKeys | vk | 3.8178¹ | 2.4836 | 1.54x¹ | 9.02x¹ → **5.86x** |
| DeviceReduce.Sum | vk + cuda-interop² | — | 0.0230 | — | **0.49x** |
| DeviceScan.InclusiveSum | vk + cuda-interop² | — | (DSL path by policy) | — | **1.06x** |
| DeviceRadixSort.SortKeys | vk + cuda-interop² | — | 0.4341 | — | **1.015x** |

² epoch 6: cuda-backend-compiled kernels launched in the vk command buffer via
`VK_NV_cuda_kernel_launch` (`LCPP_INTEROP=auto`, plan `vk_cuda_opt.md`). Throughput
rows at `LCPP_BENCH_BATCH=100`; radix floor at batch=1: 0.8168 → 0.5509 ms (−32.6 %).

¹ epoch-0 dx/vk numbers used per-op `synchronize()` timing (~100+ µs backend
round-trip per op); final numbers use amortized timing — see
`docs/epoch1_host_overhead.md`. Same-backend before/after comparisons for dx/vk
are therefore indicative; the cuda column is directly comparable.

**Headline: on CUDA, all three primitives now run at CUB parity (0.96x–1.11x),
with scan and radix ~2.4x / ~1.8x faster than at epoch 0.**

## Epochs

| epoch | focus | key change | result (cuda) | doc |
|---|---|---|---|---|
| 0 | baseline + harness | bench target, CUB wrappers, vk/dx portability fixes | — | `epoch0_baseline.md` |
| 1 | host dispatch | string shader-cache keys → O(1) pointer-identity keys (`shader_cache.h`); fixed latent KEY_ONLY shader collision | host dispatch 0.25→0.16 µs (−35%) | `epoch1_host_overhead.md` |
| 2 | scan kernel | coalesced striped I/O via shared-memory transpose (blocked register layout preserved for order-correct BlockScan); policy 256×8→256×15 | 0.186→0.079 ms (2.4x), 0.96x CUB | `epoch2_scan_kernel.md` |
| 3 | radix sort | policy 256×4→256×16 (tile 4096, 4x fewer blocks → shorter lookback, 4x smaller lookback resets) | 0.767→0.428 ms (1.8x), 1.00x CUB | `epoch3_radix_sort.md` |
| 3.5 | review fix | sizeof-aware tile clamp `radix_effective_items_v` keeps default policy DX-safe for >4-byte keys (+ ulong test) | no perf change | `epoch3_radix_sort.md` addendum |
| 4 | reduce kernel | coalesced striped `ConsumeFullTile` loads; policy 256×4→256×8; ReduceByKey decoupled to its own DX-floor tile | 0.032→0.022 ms (1.46x), 1.11x CUB | `epoch4_reduce_kernel.md` |
| 5 | vk legacy HLSL route | `lc_vk_backend_use_xir_spirv=false`: radix one-sweep 3.5x faster on the legacy HLSL/DXC vk route (0.71 ms, 1.68x CUB); gap is specific to XIR→SPIR-V codegen | — | `epoch5_vk_legacy_hlsl.md` |
| 6 | vk-cuda interop | `VK_NV_cuda_kernel_launch`: all ops dispatch cuda-compiled kernels in the vk command buffer; radix 0.690→0.434 ms (−37 %, 1.015x CUB); per-op policy `LCPP_INTEROP=auto`; LC `KernelLauncher` single-alloc patch; K-ops-per-CommandList floor lever | radix at CUB parity on vk | `epoch6_vk_cuda_interop.md` |

## Verification (final state)
- Full regression `scripts/run_regression.sh`: **40/40 green** — 21 legacy rows
  (7 tests × cuda/dx/vk, incl. 16.7M-assert scan and 67-assert radix with the ulong
  default-policy case) + vk-interop rows for all 7 tests + policy rows
  (`LCPP_INTEROP=auto`, explicit lists) with non-vacuous fast-path assertions —
  `docs/final_regression_vk_cuda_interop.log` (epoch 0–4 log preserved as
  `docs/final_regression.log`).
- Ragged sizes (1,000,003 / 65,537 items) verified exactly on cuda/dx/vk and on the
  vk-interop path (bit-exact vs host *and* vs the DSL path).
- Full `xmake build` (all targets) OK.

## What remains (documented, out of scope)
- ~~vk radix sort on the XIR→SPIR-V path (5.9x CUB)~~ **Resolved (epochs 5–6):**
  epoch 5 showed the gap is specific to the XIR→SPIR-V codegen (legacy HLSL route:
  1.68x CUB); epoch 6 closed it outright — cuda-compiled kernels in the vk command
  buffer run at **1.015x CUB** (`epoch6_vk_cuda_interop.md`). The residual vk host
  floor (~120–160 µs per commit) is only amortizable (K ops per CommandList),
  not removable without vk submit-architecture work (documented non-goal).
- dx/vk residual ratios for scan/radix partly reflect cross-backend comparison
  (CUB runs natively on CUDA on the same GPU) plus dx/vk dispatch overhead.
- `LCPP_BENCH_GRAPH=1` (CUDA graph capture) exists for submit-overhead-free
  cuda measurements.
