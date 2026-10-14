# Epoch 4 — Optimizing the device reduce kernel (DeviceReduce::Sum et al.)

## TL;DR

Coalesced the full-tile inner loop of `AgentReduceImpl::ConsumeFullTile` from a
blocked (thread-major) to a block-striped global load pattern, and tuned the
`DeviceReduce` tile policy from 256×4 to 256×8. Because the `ReduceByKey` kernel
allocates three shared arrays each sized by the tile, it has its own DX-floor-bound
tile (kept at 256×4, bit-identical to the epoch-3 baseline) so that the accelerated
default for Sum/Reduce/Min/Max/Arg* does not break DX. On 4,194,304 int32 (RTX 5060):

| backend | epoch-0/3 (256×4, blocked) | epoch-4 (256×8, coalesced) | CUB | lcpp/CUB before → after |
|---------|----------------------------|----------------------------|------|--------------------------|
| cuda    | 0.0320 ms (1.54×)          | 0.0221 ms                  | 0.0200 ms | 1.54× → **1.11×** |
| dx      | 0.0478 ms (1.02×)          | 0.0392 ms                  | 0.0451 ms | 1.02× → **0.87×** |
| vk      | 0.0458 ms (0.98×)          | 0.0365 ms                  | 0.0452 ms | 0.98× → **0.81×** |

cuda improved 1.45× (0.032 → 0.022 ms) and lands inside the epoch goal (≤ ~1.15× CUB);
dx/vk improved further below parity. The epoch goal (cuda ≤ ~1.15× CUB) is met with
coalescing + policy alone, so the optional uint4 vectorization (step 3) was **skipped**
(rationale below). Every change is justified by measurement; the full 21-combo regression
(7 tests × {cuda,dx,vk}) is green and the bench verifies the exact int Sum against both a
host reference and CUB on every backend and at the ragged 1,000,003 size (exit 0).

Measurement mode (unchanged from epochs 2/3): cuda uses CUDA events on the LuisaCompute
stream; dx/vk use the amortized wall-clock `HostSyncTimer` (batch=10 back-to-back enqueues
bracketed by one synchronize()); the CUB baseline always runs on the same physical GPU (its
own CUDA stream). Reported medians are the minimum stable run after retrying the known
transient empty-stdout flakiness (re-runs until the verified `DeviceReduce` row printed).

## Root cause and change

### 1. Coalesced `ConsumeFullTile` (`src/lcpp/agent/agent_reduce.h`)

The hot full-tile loop loaded a thread's `ITEMS_PER_THREAD` register array with
`global_index = block_offset + land_id*ITEMS_PER_THREAD + i` (i = 0..ITEMS-1): a blocked
layout where adjacent threads are `ITEMS` elements apart → poorly coalesced global loads
(mitigated only by L1/L2). Changed to a block-striped layout:

```
global_index = block_offset + i*BLOCK_SIZE + land_id;
```

so adjacent threads touch adjacent elements at each `i`, matching `ConsumePartialTile`'s
stride (which was already striped). `ConsumePartialTile` (tail path) semantics are unchanged.

**Why the reorder is exact here (unlike the scan):** reduce is commutative + associative
per thread — each thread folds its own items into one thread-aggregate and the `BlockReduce`
handles the cross-thread combine, so only the per-thread grouping matters, not which global
positions a thread happened to read (the same reordering caveat CUB states for non-commutative
ops). The tests verify exact int Sum (int addition is associative modulo 2^32, no overflow in
the bench input), and Min/Max and ArgMin/ArgMax on distinct shuffled values. The change touches
`agent_reduce.h`, not `block_reduce.h`/`warp_reduce.h`, so `block_level_test`/`warp_level_test`
public behavior is unaffected.

### 2. Policy sweep (`LCPP_REDUCE_ITEMS` × `LCPP_REDUCE_BLOCK`, 4 M int, median ms)

Shared (4-byte single-array reduce path) = `(BLOCK*ITEMS + BLOCK*ITEMS/32) * 4`.

| policy (B×I) | tile | cuda | dx | vk | 4B shared | 8B (Arg*/IVP) |
|--------------|------|------|------|------|-----------|----------------|
| 256×4 (base) | 1024 | 0.0306 (1.49×) | 0.0469 (0.99×) | 0.0436 (0.96×) | 4224 | 8448 |
| **256×8**    | 2048 | **0.0221 (1.11×)** | **0.0392 (0.87×)** | **0.0361 (0.77×)** | 8448 | 16896 |
| 256×16       | 4096 | 0.0222 (1.10×) | 0.0412 (0.88×) | 0.0346 (0.76×) | 16896 | **33792 ✗** |
| 128×8        | 1024 | 0.0244 (1.21×) | — | — | 4224 | 8448 |
| 128×16       | 2048 | 0.0221 (1.10×) | — | 0.0333 (0.71×) | 8448 | 16896 |

Notes from the sweep:

- At 256×4, coalescing alone did **not** move cuda (blocked reads were already absorbed by
  the L2-resident 4 M input); the gain came from the policy. Fewer, larger tiles cut the grid
  (4096 → 2048 blocks) and shorten the single-tile recursion tail (2048 partials → 2 coalesced
  striped passes instead of 4 blocked passes), while raising per-thread ILP.
- 256×16 ties 256×8 on cuda/vk but is rejected as the *library default* because the 8-byte
  `IndexValuePairT` used by the default-policy ArgMin/ArgMax reduce needs
  `(256*16 + 128) * 8 = 33792 B > 32768` — the DX/HLSL group-shared floor (this is a regression
  the epoch-3.5 radix clamp guards against; the reduce default must respect the same rule).
- 256×8 is the best portable policy: cuda ≤ 1.15× goal, dx/vk below parity, and DX-safe for
  both the 4-byte (8448 B) and 8-byte (16896 B) single-array reduce paths.

### 3. `DeviceReduce` default and the reduce-by-key DX floor

Baked the winner into the library default and the bench macro (they match):

- `src/lcpp/device/device_reduce.h` — new `static constexpr size_t REDUCE_ITEMS_PER_THREAD = 8;`
  used as `DeviceReduce`'s `ITEMS_PER_THREAD` template default (was `details::ITEMS_PER_THREAD = 4`).
  The Sum/Reduce/Min/Max/TransformReduce/Arg* kernels and their temp sizing follow automatically.
- `tests/lcpp_cub_bench.cpp` — `LCPP_REDUCE_ITEMS 4u → 8u` (block stays 256u). No other bench edits.

The default tile bump initially broke `device_reduce_test` on **dx** with
`Assertion 'shared_size <= 32768' failed` — but the failure was in **ReduceByKey**, not the Sum
path. `ReduceByKeyModule::compile` allocates THREE shared arrays each sized by the tile element
count (`s_keys`, `s_values` at 4 B, and the two-phase-scatter `s_scatter_keys` at 8 B), so at
256×8 it needs `(2048+64)*(4+4+8) = 33792 B > 32 KiB`, ~3× more sensitive than the single-array
Sum path. Fix: decouple reduce-by-key to its own DX-floor-bound tile.

- `src/lcpp/device/device_reduce.h` — new `static constexpr size_t REDUCE_BY_KEY_ITEMS_PER_THREAD = 4;`
  and a `m_rbk_shared_mem_size` computed in `create()`. The reduce-by-key dispatch
  (`GetReduceByKeyTempStorageBytes` num_tiles, the public `ReduceByKey` num_tiles, and
  `reduce_by_key_array`'s `tile_items` / `ReduceByKeyModule` template arg / `compile()` shared
  size) now uses this tile instead of the shared `ITEMS_PER_THREAD`. Reduce-by-key therefore keeps
  the exact epoch-3 tile, shared footprint, temp layout, and results — only the hot reduce ops move
  to 256×8. Public API and behavior are unchanged (verified: `device_reduce_test` reduce_by_key green
  on cuda/dx/vk, 71 asserts).

## Optional uint4 vectorized striped loads — SKIPPED

Step 3 was conditional on cuda staying > 1.2× CUB after steps 1+2. cuda reached 1.11×, so it is
out of scope. Independently: a vectorized 16 B striped load needs per-thread-group 4-alignment and
a `BufferView` element-type reinterpret that is risky across cuda/dx/vk alignment and ragged tails
(the same reason the analogous option was skipped in epoch 2), and the remaining absolute delta to
CUB is ~0.002 ms. Not attempted.

## Correctness / regression

- `lcpp_cub_bench cuda|dx|vk` (4 M): exact int Sum verified vs host ref and CUB, exit 0
  (cuda 0.0221, dx 0.0392, vk 0.0365 ms).
- Ragged `lcpp_cub_bench cuda|dx|vk 1000003 10`: exit 0 (exact) — exercises the striped full-tile
  range + `ConsumePartialTile` tail under the larger tile.
- Full build `xmake build` (all targets): OK.
- Full regression `scripts/run_regression.sh` — **21/21 green** (7 tests × {cuda,dx,vk}):
  block_level_test 1025, warp_level_test 592, decoupled_look_back 2, device_reduce_test 71,
  device_scan_test 16,777,208, device_segment_reduce 86, device_radix_sort_one_sweep 67.
  `device_reduce_test dx` (which exercises ReduceByKey on the DX floor) passes after the decouple.

## Summary

cuda 1.54× → 1.11× CUB (0.0320 → 0.0221 ms, 1.45× faster, epoch goal met); dx 1.02× → 0.87×;
vk 0.98× → 0.81×. All backends bit-exact and the full 21-combo regression is green.
