# Epoch 2 — Optimizing the device scan kernel (`DeviceScan::InclusiveSum` / `ExclusiveSum`)

## TL;DR

Replaced the uncoalesced per-thread **blocked** global load/store in the scan
kernel with a **coalesced block-striped I/O transposed through shared memory**,
and tuned the tile policy from `256×8` to `256×15`. On 4,194,304 `int32` items
(RTX 5060):

| backend | epoch-1 (`256×8`, DIRECT) | + transpose (`256×8`) | epoch-2 tuned (`256×15`) | CUB | speedup vs epoch-1 |
|---------|---------------------------|-----------------------|--------------------------|------|--------------------|
| cuda | 0.165 ms (2.0×)   | 0.0866 ms | **0.079 ms (≈0.98× CUB)** | 0.081 ms | ~2.1× |
| dx   | 0.182 ms (2.3×)   | 0.114 ms  | **0.100 ms (≈1.25× CUB)** | 0.080 ms | ~1.8× |
| vk   | 0.231 ms (2.9×)   | 0.109 ms  | **0.093 ms (≈1.15× CUB)** | 0.078 ms | ~2.5× |

Effective bandwidth on cuda rose from ~194 GB/s to ~405 GB/s (32 MB read+write /
0.079 ms), essentially at the card's ~450 GB/s DRAM roofline. On cuda the scan
now runs **at or slightly below CUB parity**; dx/vk are within ~1.15–1.25× (their
CUB baselines are measured on a separate CUDA stream while lcpp runs on dx/vk).

**Measurement mode:** cuda uses CUDA events on the LuisaCompute stream
(`CudaStreamTimer`). dx and vk use the amortized wall-clock `HostSyncTimer`
(`batch=10` back-to-back enqueues bracketed by one `synchronize()`, per-op
average) — the default for non-cuda backends. The `expected` scan output is
verified **exactly** against a host `inclusive_scan` in every run; the bench
exits non-zero (3) on any mismatch — it returned **0** for every measurement
below, including the ragged correctness runs.

## Root cause (confirmed)

`details::ScanModule::compile` (`src/lcpp/device/details/scan.h`) loaded/stored
through `BlockLoad`/`BlockStore` whose `BLOCK_LOAD_DIRECT`/`BLOCK_STORE_DIRECT`
path is **blocked**: thread `t` touches global elements
`[tile_start + t*ITEMS, tile_start + t*ITEMS + ITEMS)`, so adjacent threads are
`ITEMS` elements apart (32 B apart at `ITEMS=8`) → poorly coalesced global
loads/stores. This, not the warp-shuffle `BlockScan`, was the DRAM bottleneck.

## Why not a plain striped load/store (correctness)

`BlockScan` (warp-shuffle) computes the prefix in **thread-major logical order**:
thread `t`, slot `i` is logical tile position `t*ITEMS + i`. For the block output
to be the true global prefix sum, that value *must* be stored at global tile
position `t*ITEMS + i` (i.e. blocked). A pure striped load **and** matching
striped store (`i*BLOCK + t`) is **not** monotone in the logical order
(`i*BLOCK + t` decreases when moving from the last slot of one thread to slot 0
of the next), so it produces wrong results — the bench would catch this.

The fix therefore transposes through shared memory (`s_data`, already allocated
but unused in DIRECT mode): global I/O is block-striped (coalesced), while the
per-thread register array stays blocked (thread-major) so the *unmodified*
`BlockScan` and the decoupled-look-back prefix op work exactly as before.

## What changed

* `src/lcpp/device/details/scan.h` — the load and store blocks of the scan
  kernel now:
  1. coalesced-fetch tile positions `{i*BLOCK + tid}` into `s_data` (guarded by
     `p < num_remaining` on the last tile, comparing the **relative** tile
     position before adding `tile_start`, mirroring the old guarded DIRECT load);
  2. `sync_block()`;
  3. blocked-read `s_data[tid*ITEMS + i]` into the `items` register array
     (`Type4Byte(0)` identity past `num_remaining` — same default the old code
     used); run `BlockScan` / `TilePrefixCallbackOp` unchanged;
  4. blocked-write results into `s_data`; `sync_block()`; coalesced-store
     `s_data[i*BLOCK + tid]` back to `d_out` (guarded on the last tile).
  The `(tid,i) -> tile position` mapping is identical on the load and store
  sides, so element ordering is preserved end-to-end.
* **No change** to `BlockLoad`/`BlockStore`/`BlockScan` public helpers
  (`block_load.h`, `block_store.h`, `block_scan.h`) — `block_level_test` is
  unaffected and continues to pass. The existing striped helpers were left as-is
  (the guarded `BufferVar` striped-load helper has a latent `*sizeof(T)` issue,
  so the kernel inlines its own transpose rather than reuse it).
* `src/lcpp/device/device_scan.h` — `DeviceScan` default `ITEMS_PER_THREAD`
  changed from `details::ITEMS_PER_THREAD` (4) to **15** (4-byte policy;
  `BLOCK_SIZE` stays `details::BLOCK_SIZE` = 256). Shared bytes
  `(256*15 + 256*15/32)*4 = 15840 B` ≤ the DX 32 KiB floor.
* `tests/lcpp_cub_bench.cpp` — `LCPP_SCAN_ITEMS` `8u` → `15u`
  (`LCPP_SCAN_BLOCK` stays `256u`). No other bench edits.

## Policy sweep (4 M `int32`, transpose on, median; ms)

Shared bytes = `(BLOCK*ITEMS + BLOCK*ITEMS/32)*4`, all under the 32 KiB DX cap.

| policy (B×I) | shared B | cuda | dx | vk |
|--------------|----------|------|------|------|
| 256×4   | 4224  | 0.114 | —     | —     |
| 256×8   | 8448  | 0.0868 | 0.1142 | 0.1094 |
| 128×8   | 4224  | 0.0923 | —     | —     |
| 320×8   | 10560 | 0.0845 | 0.1132 | 0.1065 |
| 512×8   | 16896 | 0.0819 | 0.1021 | 0.0998 |
| 128×15  | 7920  | 0.0814 | 0.1094 | 0.1026 |
| 256×12  | 12672 | 0.0788 | 0.1060 | 0.0957 |
| **256×15** | 15840 | **0.0795** | **0.0996** | **0.0930** |

`256×15` and `256×12` are the finalists (confirmed with ≥3 clean runs each).
`256×12` is marginally best on cuda (0.079), but `256×15` is best on both dx and
vk and within ~1% on cuda, so **`256×15` is the chosen all-round policy** and is
baked into the library default and the bench macro. Larger blocks (512) and
smaller `ITEMS` (4–8) regress — coalescing benefits from more elements per tile
per thread, while look-back depth/occupancy penalize the extremes.

## Correctness / regression

* `lcpp_cub_bench cuda|dx|vk` (4 M): exact scan match (exit 0), reported above.
* Ragged, non-multiple size `cuda|dx|vk 1000003 10`: exact match (exit 0) —
  exercises the guarded tail-transpose path.
* Full regression `scripts/run_regression.sh` (21 combos = 7 tests × {cuda,dx,vk}):
  **all green** — `block_level_test` (1025 asserts), `warp_level_test`,
  `decoupled_look_back`, `device_reduce_test`, `device_scan_test`
  (16,777,208 asserts — inclusive/exclusive `DeviceScan<256,32,4>` for every
  power-of-two size 1…8 M + `scan_by_key`, i.e. the transpose validated at
  `ITEMS=4` too), `device_segment_reduce`, `device_radix_sort_one_sweep`.

## Optional items (not done — rationale)

The primary cuda target (~0.10–0.12 ms) is already met/exceeded and cuda is at
CUB parity, so the two optional follow-ups were skipped to avoid risk:
* Vectorized `uint4` striped loads via a `BufferView` cast — the DSL buffer read
  is element-typed; a 16 B vectorized path would need a view-cast that is not
  obviously safe across dx/vk/cuda alignment + ragged tails. The scalar
  block-striped transpose already reaches ~90% of the roofline on cuda, so the
  remaining vectorized headroom is small.
* Deeper `TilePrefixCallbackOp` look-back tuning — unchanged; the `tile_id==0`
  first-tile special case already exists.

## Remaining concerns

* dx/vk still trail CUB (1.15–1.25×) because their CUB baseline runs on a
  separate CUDA stream on the same GPU while lcpp pays the dx/vk command-list /
  backend dispatch path; the amortized `HostSyncTimer` (batch=10) is comparable
  but not identical to the cuda event timer. GPU-side scan time is DRAM-bound
  and close to roofline; the residual gap is backend dispatch overhead, not the
  scan kernel itself.
* The transpose adds one `sync_block()` in the load and (net) one in the store
  vs the old DIRECT path; measured cost is well outweighed by coalescing. If a
  future policy pushes `BLOCK*ITEMS` near 32 KiB on DX, revisit shared sizing.
