Epoch 3 — Optimizing `DeviceRadixSort::SortKeys` (one-sweep radix sort)

TL;DR

A per-phase GPU-time breakdown showed the **one-sweep pass kernel is ~90 % of the
total on every backend** (histogram + resets + exclusive-sum are ~0.04–0.09 ms in
aggregate). The single effective, low-risk lever was the tile policy: raising
`ITEMS_PER_THREAD` from 4 to 16 (block 256, tile = 256×16 = 4096 keys) shrinks the
grid 4× (4096 → 1024 blocks for 4 M keys), which simultaneously shortens the
decoupled-lookback chain, cuts the per-pass lookback-reset volume (4 MB → 1 MB),
and raises per-thread ILP. Adopted as the library default (`RADIX_ITEMS_PER_THREAD = 16`)
and mirrored in the bench macro. On 4,194,304 uint32 keys (RTX 5060):

| backend | epoch-2 (256×4) | epoch-3 (256×16) | speedup | CUB | lcpp/CUB before → after |
|---------|-----------------|------------------|---------|------|--------------------------|
| cuda    | 0.766 ms (1.80×) | **0.428 ms**     | 1.79×   | 0.426 ms | 1.80× → **1.01× (parity)** |
| dx      | 0.815 ms (1.92×) | **0.671 ms**     | 1.21×   | 0.424 ms | 1.92× → **1.58×** |
| vk      | 3.67  ms (8.63×) | **2.485 ms**     | 1.48×   | 0.426 ms | 8.63× → **5.83×** |

Effective one-sweep work on 4 M keys: 4 passes × (read 16 MB + write 16 MB) = 128 MB.
At the card's ~450 GB/s DRAM roofline that is ~0.28 ms, so cuda at 0.428 ms total
(histogram + resets + 4 sweeps) is essentially at roofline / CUB parity. The remaining
vk gap is **inside the one-sweep kernel** and is a backend execution characteristic of the
XIR-SPIR-V path (see “What remains”), not addressable by the policy/reset knobs.

Measurement mode (unchanged from epoch 2)

* cuda: CUDA events on the LuisaCompute stream (`CudaStreamTimer`, batch=1, no graph).
* dx / vk: amortized wall-clock `HostSyncTimer`, batch=10 back-to-back enqueues
  bracketed by one `synchronize()` (per-op average; ~0.01 ms dispatch overhead).
* CUB baseline always runs on the same physical GPU (native CUDA stream).
* Every reported run verified the sorted output **exactly** against the host sort and
  the CUB result (bench returns non-zero on mismatch; all kept runs returned 0).
* A transient empty-stdout / relaunch flakiness was observed when relaunching the bench
  in rapid succession (no log line, `exit=0`); measurements below are from runs that
  printed the verified radix row (retried until a line appeared).

Phase breakdown (method)

Differential stage-skipping in a throwaway copy of the pipeline driver: each phase was
gated out by an env flag and `lcpp_cub_bench` rebuilt once; per-phase GPU time was
recovered by subtraction. Removing a *producer* phase (histogram / exclusive-sum /
init-reset) while keeping the scatter consumer writes out-of-bounds (garbage digit
offsets), so those phases were isolated **with the one-sweep also skipped** (no scatter
⇒ no OOB); the one-sweep cost is `full − (everything-except-sweep)`. All experiments
were reverted; nothing experimental is committed.

Phase breakdown at the epoch-2 baseline policy (256×4), median ms:

| phase                              | cuda | dx   | vk   |
|------------------------------------|------|------|------|
| **one-sweep × 4 passes**           | 0.69 | 0.73 | 3.58 |
| histogram                          | 0.023| 0.042| 0.047|
| lookback reset (1 M uint × 4 passes)| ~0.019| 0.019| 0.021|
| exclusive-sum of digit counts      | ~0   | 0.002| ~0   |
| init reset (bins + ctrs)           | small| small| small|
| **total**                          | 0.77 | 0.82 | 3.67 |

Conclusions that drove the change:

1. **Hypothesis (a) — VK outlier — confirmed and localized**: the vk 8.6× is ~entirely
   the one-sweep kernel (3.58 ms of 3.67 ms). Histogram/reset/exclusive-sum on vk are the
   same tiny cost as cuda/dx.
2. **Hypothesis (b) — histogram grid config — rejected**: at 256 blocks × 256 threads the
   histogram is only 0.02–0.05 ms (< 6 % of total). Not a bottleneck; left unchanged.
3. **Hypothesis (c) — lookback reset — rejected as a separate change**: ~0.02 ms (< 5 %).
   Note the adopted larger tile *also* shrinks it automatically (max_num_blocks 4096→1024
   ⇒ lookback buffer 4 MB→1 MB per pass), so a dedicated partial-reset was unnecessary.
4. **Hypothesis (d) — policy sweep — the winner** (below).
5. **Hypothesis (e) — RADIX_BITS — not pursued**: 8 bits / 4 passes is the standard
   traffic-vs-overhead point; changing it (5 or 4 passes) alters temp layout and pass count
   without addressing the measured onesweep-bound cost, so it was out of scope/risk.

Policy sweep (full pipeline, median ms; “—” = did not run to completion on that backend)

Shared bytes ≈ `BLOCK*ITEMS*sizeof(k)` (scattered keys) + `(BLOCK/32)*RADIX_DIGITS*4`
(rank warp-offsets) + `2*RADIX_DIGITS*4` (global offsets + match masks); uint32, RADIX_DIGITS=256.

| policy (B×I) | tile | approx. shared | cuda  | dx    | vk    |
|--------------|------|----------------|-------|-------|-------|
| 256×4  (base)| 1024 | ~13 KiB        | 0.772 | 0.817 | 3.68  |
| 256×8  | 2048 | ~17 KiB | 0.511 | 0.621 | 2.99 |
| 256×12 | 3072 | ~22 KiB | 0.445 | —     | 2.97 |
| **256×16** | **4096** | **~26 KiB** | **0.435** | **0.672** | **2.49** |
| 256×20 | 5120 | ~30 KiB | — | — | 2.39 |
| 128×4  | 512  | ~ 9 KiB | 1.35  | —     | —     |
| 128×8  | 1024 | ~ 9 KiB | 0.83  | —     | 4.13  |
| 128×16 | 2048 | ~14 KiB | 0.555 | 0.697 | 3.54  |
| 128×32 | 4096 | ~22 KiB | 0.496 | —     | 2.84  |
| 192×16 | 3072 | ~20 KiB | 0.537 | 0.704 | —     |
| 512×4  | 2048 | ~25 KiB | 0.679 | 0.782 | 2.97  |
| 512×8  | 4096 | ~33 KiB | —     | —     | 2.78  |

Reading the sweep:

* Larger tiles monotonically help on all backends up to the point where the block count
  becomes too small to fill the GPU / shared budget is exhausted. cuda and vk both peak
  around **256×16** among configs that run on **all three** backends.
* The configs that beat 256×16 on vk (256×20 → 2.39, 512×8 → 2.78) are **not portable**:
  they do not run to completion on cuda/dx (256×20/512×8 push the scattered-key +
  rank shared footprint past the DX 32 KiB HLSL floor, and fail on cuda in the sweep).
* 256×16 keeps DX shared (~26 KiB) under the 32 KiB floor — verified by the dx run (0.672)
  and by the full DX regression below.
* 256×12 was marginally better on cuda (0.435 vs 0.445 noise) but clearly worse on vk
  (2.97 vs 2.49), so 256×16 is the chosen single portable policy.

What changed

* `src/lcpp/device/device_radix_sort.h` — added a radix-specific default
  `static constexpr size_t RADIX_ITEMS_PER_THREAD = 16;` and set it as the
  `DeviceRadixSort` template’s `ITEMS_PER_THREAD` default (was the shared
  `details::ITEMS_PER_THREAD` = 4). This is the **only** library change: the pipeline
  structure (reset → histogram → exclusive-sum → 4 one-sweep passes) is untouched, so the
  public API and behavior are unchanged; only the tile shape (and therefore grid size,
  lookback buffer size, and temp-storage sizing derived from `ITEMS_PER_THREAD`) moves.
  The comment documents the tuning. Because `GetSortKeysTempStorageBytes` /
  `GetSortPairsTempStorageBytes` already scale with `ITEMS_PER_THREAD`, the 1-element dummy
  value-view and all temp carvings stay correct for the new tile.
* `tests/lcpp_cub_bench.cpp` — `LCPP_RADIX_ITEMS 4u → 16u` (block stays 256), so the fixed
  bench instantiates `DeviceRadixSort<256, 32, 16>` and measures the library default. No
  other bench edits. (The bench macros match the new library default.)

What remains (vk) and why

After the policy change the vk one-sweep is still ~5.7× the cuda one-sweep for the same
128 MB of traffic, while a lookback-free, dispatch-overhead-amortized kernel (the epoch-2
scan) reaches vk/CUB ≈ 1.15×. The distinguishing features of the one-sweep kernel are the
**decoupled-lookback volatile global-memory spin polling** (`d_lookback.volatile_read` in a
`while(value==0)` loop plus `volatile_write`) and the **per-key `MatchAny` warp-match** in
`BlockRadixRankMatchEarlyCounts`. On the XIR→SPIR-V path these map to conservative
(volatile/subgroup) instructions whose cost per iteration is much higher than on the native
CUDA path, so the same kernel is several times slower on vk — a backend codegen/execution
characteristic, not a policy knob. The policy change is exactly the lever that reduces the
*number* of blocks and therefore the total lookback-chain length and reset volume, which is
why vk improved 1.48× (3.68→2.49). Further vk gains would require reworking the lookback
memory mechanism / rank-match codegen itself (large, risky, out of scope here) or a
lower-radix higher-pass tradeoff that does not pay off in the breakdown.

Caveat on larger key types: the default tile (BLOCK×ITEMS) shared footprint scales with
`sizeof(key)`. 256×16 is sized for the ≤ 32 KiB DX floor for **4-byte** keys (uint32/int32/
float). For 8-byte keys (int64/uint64/double) the default exceeds the DX floor; callers on
DX should instantiate an explicit smaller policy (e.g. `DeviceRadixSort<256, 32, 8>` or
`<128, 32, 16>`) for those key types. The existing uint32 correctness test (`DeviceRadixSort<>`
default path) and the float/uint32 pair tests (which use explicit `128×4`) are unaffected.

Correctness / regression

* `lcpp_cub_bench cuda|dx|vk` (4 M): exact radix match, exit 0 — cuda 0.428, dx 0.671,
  vk 2.485 ms.
* Ragged `lcpp_cub_bench {cuda|dx|vk} 1000003 10`: exact match, exit 0 (exercises the
  guarded tail path with the larger tile; last tile < 4096 keys).
* `device_radix_sort_one_sweep` (ascending/descending uint32 sizes incl. ragged, float
  keys via `128×4`, and SortPairs): **66 asserts, all pass** on cuda/dx/vk.
* Full `xmake build` (all targets): OK.
* Full regression `scripts/run_regression.sh` — **21/21 green** (7 tests × {cuda,dx,vk}):
  block_level_test 1025, warp_level_test 592, decoupled_look_back 2, device_reduce_test 71,
  device_scan_test 16,777,208, device_segment_reduce 86, device_radix_sort_one_sweep 66.

  Summary vs baseline: cuda reached CUB parity (1.80× → 1.01×); dx 1.92× → 1.58×; the vk
  outlier 8.63× → 5.83×. All backends remain bit-exact and the full 21-combo regression is green.

  ## Addendum (epoch 3.5): sizeof-aware tile clamp (review fix)

  Review of epoch 3 found that the new default `ITEMS_PER_THREAD = 16` breaks the
  32 KiB DX shared floor for **8-byte keys** (a default `DeviceRadixSort<>` with
  uint64 keys on dx would fail at shader compile time — a regression vs epoch 0's
  default of 4). Fixed by `radix_effective_items_v<NominalItems, KeyType>`
  (`src/lcpp/agent/policy.h`): the nominal tile is kept for `sizeof(KeyType) <= 4`
  and scaled by `4/sizeof(KeyType)` (floor 4) for wider keys, applied consistently
  in `GetSortPairsTempStorageBytes` and in `onesweep_radix_sort` (tile sizing +
  histogram module). Verified by a new `ulong` default-policy test case in
  `device_radix_sort_one_sweep` (67 asserts, green on cuda/dx/vk); uint32 timings
  are unchanged (0.428 ms cuda).
