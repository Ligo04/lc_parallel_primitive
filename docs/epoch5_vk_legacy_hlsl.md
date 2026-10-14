# Epoch 5 — Vulkan legacy-HLSL (DXC) codegen path support

Date: 2026-10-14
Configuration: `xmake f -p windows -a x64 -m release --lc_vk_backend_use_xir_spirv=false`
(native XIR→SPIR-V codegen off → shaders compile through the legacy HLSL→DXC→SPIR-V
route; `lc_vk_backend_enable_dxc_compatibility` stays on as the fallback layer).

## Why

Epochs 0–4 ran the vk backend through the native XIR→SPIR-V path
(`lc_vk_backend_use_xir_spirv=true`), which was required at the time because lcpp's
temp-buffer *subviews* violated the legacy path's stricter descriptor contract.
This epoch makes the legacy HLSL path work too, so both codegen routes are usable.

## Failure enumeration (legacy path, before fixes)

Full regression with the legacy path (release): 17/21 green; 4 failures, all the
same root cause — `validate_direct_storage_buffer_descriptor`
(`LuisaCompute/src/backends/vk/resource_barrier_contract.h`) rejects buffer views
whose byte offset is not a multiple of the device
`minStorageBufferOffsetAlignment` (16 B on this GPU) or that are empty/misaligned
elements — the legacy HLSL ABI cannot express a descriptor-relative subview bias
(fail-closed assert at `src/backends/vk/stream.cpp:1181`):

| test | failing view | source |
|---|---|---|
| device_reduce_test (vk) | `[8, 12)` | `reduce_array_recursive` recursion: next-level temp region starts at `num_tiles` elements — arbitrary byte offset |
| device_scan_test (vk) | `[132, 264)` | scan/scan-by-key temp carving: `tile_partial` right after `tile_status` (33 uints = 132 B) |
| device_radix_sort_one_sweep (vk) | `[5380, 5384)` | per-pass counter subview `d_ctrs_view.subview(portion*num_passes+pass, 1)` — 4-byte offsets |

## Fix: descriptor-aligned temp-buffer carving (backend-agnostic)

New helpers in `src/lcpp/common/utils.h`:

```cpp
static constexpr size_t descriptor_view_alignment = 256;   // >= any known minStorageBufferOffsetAlignment,
                                                           // multiple of every lcpp element stride
static constexpr size_t descriptor_view_alignment_uints = descriptor_view_alignment / sizeof(uint);
inline size_t align_up_view(size_t byte_offset);           // align_up_uint(..., 256)
```

256 B is a multiple of 4/8/16-byte element strides, so it simultaneously satisfies
the offset-alignment, whole-element and (still) the pair-struct `sizeof` carving
rules from epoch 0/3.5. Applied uniformly — a no-op for cuda/dx, negligible padding
(a few hundred bytes per region):

- **`utils.h` `get_temp_size_scan`** now walks *byte* offsets, aligning every
  recursion-level region (new `elem_size` parameter; all 7 call sites in
  `device_reduce.h` pass `sizeof(Type4Byte)`/`sizeof(IVP)`), and
  `reduce_array_recursive` aligns its `offset` argument at entry
  (`static_assert(descriptor_view_alignment % sizeof(Type) == 0)`).
- **`device_reduce.h`** — ArgMin/ArgMax + ReduceByKey carvings aligned to 256 B.
- **`device_scan.h`** — all 8 carving walks (plain scan ×2, scan-by-key ×2 and the
  two static size functions) aligned to 256 B.
- **`device_radix_sort.h`** — lookback/tmp2/ctrs regions aligned; the per-pass
  counter now owns a dedicated 256-byte slot (`ctrs_count = portions*passes*64`
  uints, subview index `(portion*num_passes+pass)*64`), mirrored exactly in
  `GetSortPairsTempStorageBytes`.
- **`device_segment_reduce.h`** — Arg temp carvings (`d_out_kv` base) aligned.

Public APIs, temp-storage *callers* (sizes grow by <1 KiB — transparent), kernel
code and results are unchanged.

## Results after the fix (legacy vk path, 4M items, RTX 5060)

Full regression: **21/21 green** (`docs/final_regression_legacy_hlsl.log`), incl.
ragged-size verification on the bench.

| op | vk legacy HLSL | vk XIR-SPIR-V (epoch 3/4) | CUB | lcpp/CUB |
|---|---|---|---|---|
| DeviceReduce.Sum | 0.0346 ms | 0.0362 ms | 0.0469 | **0.74x** |
| DeviceScan.InclusiveSum | 0.0927 ms | 0.0921 ms | 0.0845 | 1.10x |
| DeviceRadixSort.SortKeys | **0.7139 ms** | 2.4836 ms | 0.4243 | 1.68x |

Notable finding: the epoch-3 conclusion that the vk radix gap was "inherent to the
kernel" was an artifact of the XIR→SPIR-V route — through the legacy HLSL path the
same one-sweep kernel is **3.5x faster** (0.71 vs 2.48 ms), i.e. the volatile
look-back polling / MatchAny rank codegen on the XIR→SPIR-V path was the real cost.
cuda/dx numbers are unchanged by the carving alignment (cuda 4M: reduce 0.0224,
scan 0.0801, radix 0.4333 ms — within noise of the epoch-4 finals).

## Recommendation

Both vk codegen routes now pass the full suite. The native XIR→SPIR-V path remains
the better default for radix-sort-heavy workloads *unless* its one-sweep codegen
improves; the legacy HLSL path is fully supported as the compatibility fallback.
