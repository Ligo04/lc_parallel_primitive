# Epoch 6 — vk↔CUDA interop (`VK_NV_cuda_kernel_launch`)

Date: 2026-10-14
Hardware: NVIDIA GeForce RTX 5060 (8 GB, Blackwell sm_120), Windows 11 / WDDM, CUDA 13.4
Benchmark: `tests/lcpp_cub_bench.cpp` vs nvcc-compiled CUB baseline, 4,194,304 items,
median of 10 reps, exact verification against a host reference on every run.
Build: xmake, mode=release, `lc_vk_cuda_interop=true` (LUISA_VULKAN_ENABLE_CUDA_INTEROP).
Plan: `vk_cuda_opt.md` (kept out of git); per-epoch subagent reports in
`.kimix_cache/epoch_{A,B,C,D}_report.md`.

## Headline

The vk GPU-side codegen gap is closed by launching **cuda-backend-compiled DSL kernels
inside the Vulkan command buffer** through LuisaCompute's `VkCudaInterop`
(`vkCmdCudaLaunchKernelNV`). Radix sort — the one op with a real SPIR-V codegen gap —
went from **0.690 ms (1.61× CUB) to 0.434 ms (1.015× CUB), −37 %**, landing exactly on
the cuda-backend column. Every op is bit-exact on every path; the full regression is
**40/40 green** (21 legacy cuda/dx/vk + vk-interop rows incl. ragged 1,000,003 / 65,537).

| op (vk, 4M int32, batch=100) | vk DSL (epoch 5 final) | vk interop | cuda backend | CUB same-run |
|---|---|---|---|---|
| DeviceRadixSort.SortKeys | 0.6898 ms (1.61×) | **0.4341 ms (1.015×)** | 0.4285 | 0.4260 |
| DeviceScan.InclusiveSum | 0.0852 (1.06×) | 0.0873 (1.09×) → policy: DSL | 0.0774 | 0.0811 |
| DeviceReduce.Sum | 0.0230 (0.49×) | 0.0230 (0.49×) | 0.0222 | 0.0468 |

batch=1 host floor: radix 0.8168 → 0.5509 ms (−32.6 %, 12 launches × cheaper
per-launch encode); the ~120–160 µs per-commit round-trip floor itself is unchanged
(documented non-goal: no vk submit-architecture changes).

## Design

`src/lcpp/runtime/interop_dispatch.h` — `InteropContext`:
- Feature-detects at create(): `VkCudaInterop` extension present,
  `cuda_kernel_launch_supported()`, `cuda_device_index() >= 0` (LUID match), cuda
  backend installed. Any failure → one `LUISA_WARNING`, nullptr → the pure-vk DSL path
  (graceful degradation is the default, plan risk R3).
- Owns the paired cuda `Device` (compile/import only — never dispatches) and imports
  shaders via `create_cuda_kernel(shader)` into a move-only RAII cache keyed by the
  existing `ShaderCacheKey` (R6: modules — destroyed before the vk device — own the
  imported handles).
- Cache miss: compile the *same* DSL lambda on the cuda device → import → cache.
  Call sites swap only the shader object; usages are compile-time `CudaArg<T, Usage>`
  in the typed signature (R2); host-side code (temp carving, DoubleBuffer, lcpp_check)
  untouched. Mixed lists (interop launches + ordinary vk copies in one CommandList)
  are ordered by the existing command-reorder pass.

**R1 (dynamic shared memory) — resolved, no LC patch needed.** The cuda backend emits
*static* `__shared__ lc_aligned_storage<...>` for every `Shared<T>{n}` in lcpp
(host-constant sizes at trace time); corpus-verified over 150 PTX modules
(0 `extern .shared`; static footprints 4 B–20 KB incl. radix one-sweep and scan tiles).
All launches use the thread-count typed dispatch (sharedMemBytes = 0).

## Epochs

| epoch | scope | outcome |
|---|---|---|
| A | feasibility spike: `InteropContext` + DeviceReduce plain path, R1 gate, A/B measurement | ABI proven end-to-end; bit-exact 5 sizes × 4 ops; interop == cuda column (1.05×) at batch≥100. Calibration finding: reduce is **not** discriminating — the plan's "vk 0.036 vs cuda 0.022" gap was a batch-10 sync-amortization artifact; only radix/scan had a real GPU-side gap. Recording cost +22 % vs DSL (root cause: `KernelLauncher` allocs). |
| B | full op coverage: Scan (+ByKey), SegmentReduce (+3 Arg kernels), ArgMin/ArgMax (struct-buffer + by-value-struct ABI, 16-byte slots asserted), ReduceByKey (mixed launch+copy lists), RadixSort (4 kernel families, 12 launches) — radix last | All bit-exact (64 checks) vs host *and* vs the DSL path; radix **0.6873 → 0.4329 ms (1.59×), 1.015× CUB** — the codegen hypothesis confirmed exactly where predicted (one-sweep look-back + MatchAny rank). Scan surprise: interop 2.6 % *slower* (later traced in C). Regression extended to a vk-interop matrix with non-vacuous R3 assertions (exit 3 if a module silently falls back). |
| C | host-side throughput: bench knobs, poll completion, K-ops-per-CommandList, recording-cost fix, scan-gap investigation | Three plan hypotheses **refuted with measurements**: throttle-off is catastrophic at batch=100 (+4756 % reduce — WDDM submission-queue pathology; the depth-2 throttle is load-bearing); poll mode adds a submit and is slower (user-event completion is published by the same worker thread); the vkGetBufferDeviceAddress/encode hypothesis is refuted (interop commit cost is 1.3–2× *cheaper* per op). **K ops per CommandList** is the real floor lever: reduce −65 %, scan −46 %, radix −21 % per op at K=8. LC patch (1 header, `vk_cuda_interop.h`): single-allocation `KernelLauncher` + `reserve()` → encoder gap +40→+24 ns/dispatch (−40 %). **Decision: `LCPP_INTEROP=auto`** — reduce/segment/radix interop, scan DSL (scan +2.42 % → +0.15 %, radix keeps −37 %). Regression 39/39. |
| D | policy re-tune under CUDA codegen | Radix tile sweep 8/16/24/32/40: **256×16 already optimal** for the cuda codegen (32 = −0.5 %, a wash; and the vk SPIR-V path shares the DX 32 KiB HLSL floor — discovered this epoch). Rejected. Reduce 256×8→256×16 **adopted, gated** to the interop path, integral ≤4-byte, identity-transform only (`REDUCE_INTEROP_ITEMS_PER_THREAD=16`; ArgMin/ArgMax's 8-byte tie-breaking and FP non-associativity keep the portable tile; DSL/cuda/dx bit-identical by construction): tie at 4M, **−1.45 % at 16M** (non-overlapping medians). Review found + fixed a latent mixed-geometry hazard on the fail-closed fallback call; verified by import-failure injection. Regression **40/40**. |

## Policy / usage

- `LCPP_INTEROP` env (vk only): `auto` (default recommendation) | `1`/`all` |
  `reduce,scan,segment,radix` list | `0`/`none`. Both dispatch paths always exist;
  the DSL vk path remains the reference implementation.
- Bench knobs (see `--help` / README): `LCPP_BENCH_BATCH` (1 = host floor,
  ≥100 = throughput — throughput tables must be taken at N≥100), `LCPP_BENCH_CL_OPS=K`
  (K non-aliasing ops per CommandList, all results exact-checked), `LCPP_BENCH_POLL`,
  `LCPP_BENCH_INQUEUE_LIMIT` (both refuted as wins, kept as opt-in diagnostics).
- New tests: `vk_cuda_interop_reduce_test` (R1/R1b/R3 gates, exactness),
  `vk_cuda_interop_ops_test` (87 per-op checks), `policy_tune` (tile sweep harness),
  policy-aware R3 guards in all device tests + probe, regression rows for
  interop=1/auto/list.

## LuisaCompute patch list (uncommitted, separate repo)

Exactly one file: `include/luisa/backends/ext/vk_cuda_interop.h` (+127/−25) —
single-allocation `KernelLauncher` (`_blob` of [Argument records][uniform payload],
offset-tracked), public `reserve()` called with compile-time counts from the typed
launch (3 allocs instead of ~11, no second copy), growth/fix-up preserved for
hand-rolled launchers. Additive only; ABI verified by an in-tree packing self-check
mirroring LC's `test_vk_cuda_launch_command`. Diff: `.kimix_cache/epochC/lc_vk_cuda_interop.diff`.
No vk submit-architecture changes (explicit non-goal, unchanged).

## Plan risks — final status

| # | risk | status |
|---|---|---|
| R1 | dynamic shared memory | resolved: static `__shared__` only, corpus-verified |
| R2 | wrong Usage → barrier-free layer merges a hazard | typed compile-time usages, per-kernel reviewed; K-per-CL regression rows double as the merged-layer hazard check; all exact |
| R3 | import fails closed, silently disabling fast path | non-vacuous: tests exit 3 / probe asserts when a policy-selected module reports `interop_active()==false` |
| R4 | two VA spaces on Windows | never arises: all launches go through vk (cuda device is compile/import-time only) |
| R5 | block size / geometry mismatch | imported shader records compiled block size; `set_block_size(256)` everywhere |
| R6 | CudaShader RAII lifetime | move-only cache in module, destroyed before the vk device (documented, enforced by declaration order) |
| R7 | WSL / non-NVIDIA / missing extension | same graceful-degradation gate; identical skip semantics |
| R8 | DSL-vk vs interop-vk drift | both paths bit-exact vs host in every regression row |

## Measured negatives (kept for the record)

- Radix tiles 8/24/32/40 vs 16: +17 % / +5.7 % / −0.5 % / +30 % — 16 stays.
- `inqueue_buffer_limit=false`: no effect at batch=1, up to +4756 % at batch=100.
- `LCPP_BENCH_POLL=1`: +1–3 % slower at batch=1, neutral at batch=100.
- Interop per-dispatch recording stays ~+8 ns/arg above DSL (structural: 3 vs 2
  allocations, 32-byte Argument records; parity is the ceiling without changing the
  command ABI). Irrelevant at op level (≤ +0.05 % of any op's time).

## Pre-existing issues found (documented, not campaign scope)

- Single-tile (`n ≤ 2048`) TransformReduce compiles the terminal kernel with
  `IdentityOp` — a non-identity transform is ignored on both paths (epoch A).
- `DeviceSegmentReduce::Min/Max` helpers don't instantiate (overloaded
  `compute::min/max` defeats ReduceOp deduction); kernels verified via
  `Reduce(..., MinOp/MaxOp(), ...)` (epoch B).
- vk `Buffer::copy_to(span)` segfaults on partial destinations whose byte size isn't
  16-aligned; full-size readbacks fine (epoch B).
- Reduce terminal `copy_from(temp)` only worked for an exactly-1-element view;
  now `subview(0, 1)` (epoch D, latent-bug fix enabler).
- Regression row `vk_cuda_interop_reduce_test vk` was passing vacuously (no
  `LCPP_INTEROP` ⇒ DSL path ⇒ self-skip); now runs =1 plus an =auto row (epoch D).

## What remains (out of scope, documented)

- The ~120–160 µs per-commit vk host floor (worker-thread hand-off, timeline
  semaphores, depth-2 throttle) — only amortizable (K ops per CommandList), not
  removable without the submit-architecture work this campaign excluded.
- vk scan stays on SPIR-V by policy (+2.4 % vs interop in the one-commit-per-op
  pipeline regime only; the gap disappears when commits amortize).
- Graph capture/replay on vk (externalization seam documented in the plan §3.3).

## Verification (final)

- `scripts/run_regression.sh`: **40/40 passed, 0 failed** — 21 legacy rows
  (cuda/dx/vk) + interop rows for all 7 tests + policy rows (auto, explicit list) +
  dedicated gates; log: `docs/final_regression_vk_cuda_interop.log`.
- Bit-exactness: interop == DSL == host at 4,194,304 / 1,000,003 / 65,537 (and
  2048/2049/4096 for reduce branch structure), every op, both policies.
