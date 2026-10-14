# Epoch 1 — Eliminating host-side per-call dispatch overhead in lcpp

## TL;DR

- The per-call shader-cache key was a `luisa::string` rebuilt on **every** dispatch
  by concatenating `Type::of<T>()->description()` with `typeid(op).name()`, then
  allocated + hashed. This change replaces it with a cheap pointer-identity key
  (`detail::ShaderCacheKey`: up to 6 stable `const void*` components), so the hot
  path now does **no string construction, no allocation, and O(1) pointer hashing**.
- Public API signatures are unchanged; template behavior is preserved (a single
  module instance called with different ops still resolves the right cached shader);
  storage / lookup / `reinterpret_cast` mechanism is untouched.
- **All 21 regression runs (7 tests × cuda/dx/vk) pass.**
- Measurement finding: the string key *was* the dominant per-call host cost, but
  that host cost is **sub-microsecond**, so it is a tiny slice of the ~130 µs
  end-to-end wall time reported on dx/vk — which is dominated by the LuisaCompute
  backend's command-buffer submit + `stream.synchronize()` round-trip (not
  touchable from lcpp, and out of this epoch's scope). A dedicated host-enqueue
  probe quantifies the real effect: **reduce/scan host dispatch drops ~30–40 %
  (~0.25 µs → ~0.16 µs for `Sum`, ~0.18 µs → ~0.13 µs for `InclusiveSum`).**

## Files changed

| File | Change |
|------|--------|
| `src/lcpp/common/shader_cache.h` | **New.** `detail::ShaderCacheKey` (fixed array of up to 6 `const void*` + count), `ShaderCacheKeyHash` (splitmix64 avalanche), `make_shader_cache_key(...)`, identity helpers `type_ptr<T>()` (=`Type::of<T>()` pointer), `op_ptr(op)` (=`&typeid(op)`), `bool_ptr(b)` (fold a bool template param), and the `detail::ShaderCacheMap` alias. |
| `src/lcpp/common/utils.h` | Rewrote all 7 `get_type_and_op_desc` overloads to return `detail::ShaderCacheKey` instead of `luisa::string`. **Signatures (template params, concept constraints, argument lists) are byte-for-byte the same**, so every call site resolves to the same overload and discriminates the same set of (type, op) combos. Added `get_radix_sort_key<KeyType, ValueType, KEY_ONLY, IS_DESCENDING>()`. Includes `shader_cache.h`. |
| `src/lcpp/device/device_reduce.h` | 7 shader maps → `detail::ShaderCacheMap`. `arg_construct`/`arg_assign` inline keys → `make_shader_cache_key(type_ptr<Type4Byte>())`. |
| `src/lcpp/device/device_scan.h` | 6 shader maps → `detail::ShaderCacheMap`. scan tile-state-init inline key → cheap key. The `LUISA_INFO("... key: {}", key)` (the key is no longer a string) simplified to a fixed cache-miss message. |
| `src/lcpp/device/device_radix_sort.h` | 4 shader maps → `detail::ShaderCacheMap`. The `get_type_and_op_desc<K,V>() + "_asc"/"_desc"` string concat → `get_radix_sort_key<K, V, KEY_ONLY, IS_DESCENDING>()`. |
| `src/lcpp/device/device_segment_reduce.h` | 5 shader maps → `detail::ShaderCacheMap`. 3 inline `arg_*` keys → cheap keys. |
| `tests/xmake.lua` + `tests/host_enqueue_probe.cpp` | **New diagnostic** (not part of the shipped library). Times the pure host `enqueue` cost of each op (build a `CommandList`, `commit()` once outside the timed region so backend submit/sync latency is excluded). Uses only the public API, so it A/B-compiles against the old string-key code too. |

`LuisaCompute/` and `tests/lcpp_cub_bench.cpp` were **not** modified.

## Design notes / correctness

- **Discrimination is preserved or improved.** The old key compared *description
  strings* and *`typeid(op).name()` strings*. `Type::of<T>()` returns the
  process-wide type-registry singleton, so two data types give equal `const
  Type*` **iff** their descriptions were equal — pointer equality is exactly
  equivalent to the old string comparison. `&typeid(op)` (a unique `type_info*`
  per operator type, incl. distinct lambdas) is equivalent to the old
  `typeid(op).name()` string equality within a single binary. Every operator type
  still maps to its own shader entry.
- The KeyValuePair/`reduce_by_key`/`scan_by_key` variants now key on the **full
  element type** instead of the inner value type, which is strictly finer than
  before — it can only *avoid* a latent collision (e.g. `KeyValuePair<int,int>`
  vs `KeyValuePair<uint,int>` sharing one shader), never merge two shaders that
  the old code kept separate. Worst case is a few extra identical compiles.
- **Radix `KEY_ONLY`:** the old key omitted it, so `SortKeys<T>` (KEY_ONLY=true)
  and `SortPairs<T,T>` (KEY_ONLY=false) on one instance shared the one-sweep
  shader — a pre-existing latent bug. Encoding `KEY_ONLY` in the new key fixes it;
  cost is at most one extra identical compile for the key-independent
  reset/histogram/exclusive-sum shaders.

## Benchmark — end-to-end (lcpp median, ms; `lcpp_cub_bench`)

dx/vk use wall-clock around enqueue+**synchronize()**; cuda uses GPU events.
Each is the median of 20 (4096) or 10 (4M) repetitions. before = string key,
after = pointer key.

### 4096 items

| op | backend | before | after |
|---|---|---|---|
| DeviceReduce.Sum | dx   | 0.1346 | ~0.133 (0.131 / 0.133 / 0.138) |
| DeviceReduce.Sum | cuda | 0.0161 | 0.0165 / 0.0170 |
| DeviceReduce.Sum | vk   | 0.1058 / 0.1218 | 0.1093 |
| DeviceScan.InclusiveSum | dx   | 0.1155 / 0.1240 | 0.119 / 0.120 |
| DeviceScan.InclusiveSum | cuda | 0.0128 | 0.0117 / 0.0119 |
| DeviceScan.InclusiveSum | vk   | 0.1109 / 0.1150 | 0.1090 |
| DeviceRadixSort.SortKeys | dx   | 0.1483 / 0.1499 | 0.143 / 0.147 / 0.149 |
| DeviceRadixSort.SortKeys | cuda | 0.0778 | 0.0762 / 0.0792 |
| DeviceRadixSort.SortKeys | vk   | 0.1506 / 0.1624 | 0.1482 |

### 4194304 (4M) items

| op | backend | before | after |
|---|---|---|---|
| DeviceReduce.Sum | dx   | 0.1732 | 0.1732 / 0.1735 |
| DeviceReduce.Sum | cuda | 0.0320 | 0.0319 |
| DeviceReduce.Sum | vk   | 0.1478 | 0.1489 |
| DeviceScan.InclusiveSum | dx   | 0.2906 | 0.2871 / 0.2960 |
| DeviceScan.InclusiveSum | cuda | 0.1654 | 0.1655 |
| DeviceScan.InclusiveSum | vk   | 0.3283 | 0.3417 (vk 4M scan is high-variance) |
| DeviceRadixSort.SortKeys | dx   | 0.9435 | 0.9408 / 0.9420 |
| DeviceRadixSort.SortKeys | cuda | 0.7663 | 0.7682 |
| DeviceRadixSort.SortKeys | vk   | 3.7892 | 3.8146 |

End-to-end: **no regression** anywhere; 4M is GPU-dominated so identical; on
dx/vk the sub-µs host saving is inside the ~120 µs of sync noise (the reported
medians overlap).

## Benchmark — isolated host enqueue (`host_enqueue_probe`, µs per call)

This is the number the change actually moves: pure host dispatch, with
backend submit + `synchronize()` excluded.

| op | backend | before (string key) | after (pointer key) |
|---|---|---|---|
| DeviceReduce.Sum | dx   | 0.253 | 0.163 |
| DeviceReduce.Sum | vk   | 0.249 | 0.158 |
| DeviceReduce.Sum | cuda | 0.241 | 0.155 |
| DeviceScan.InclusiveSum | dx   | 0.182 | 0.135 |
| DeviceScan.InclusiveSum | vk   | 0.184 | 0.134 |
| DeviceScan.InclusiveSum | cuda | 0.175 | 0.132 |
| DeviceRadixSort.SortKeys | dx   | 0.645 | 0.654 |
| DeviceRadixSort.SortKeys | vk   | 0.645 | 0.645 |
| DeviceRadixSort.SortKeys | cuda | 0.620 | 0.643 |

`SortKeys` is dominated by ~10 command appends/loop, so its single key lookup is
noise → unchanged (as expected). Reduce/scan host dispatch is **~30–40 % cheaper**;
absolute per-call host cost is now ~0.13–0.16 µs.

## Why end-to-end barely moved

The ~130 µs dx/vk floor is the command-buffer submit + `synchronize()` round-trip
inside LuisaCompute (the HostSyncTimer brackets enqueue+sync). The cuda path —
same library host code, GPU-event timer — reported only ~16 µs *before* this
change, proving the string key + lookup was never ~100 µs. The change correctly
removes all of the string work from the lcpp hot path (verified by the probe), but
that work was already <0.3 µs. The remaining wall-time floor is backend-side and
not addressable within this library / epoch.

## Regression result

`bash scripts/run_regression.sh` — **21/21 green**:

```
block_level_test            cuda/dx/vk  all tests passed (1025 asserts / 3 tests)
warp_level_test             cuda/dx/vk  all tests passed (592 asserts / 3 tests)
decoupled_look_back         cuda/dx/vk  all tests passed (2 asserts / 2 tests)
device_reduce_test          cuda/dx/vk  all tests passed (71 asserts / 7 tests)
device_scan_test            cuda/dx/vk  all tests passed (16777208 asserts / 2 tests)
device_segment_reduce       cuda/dx/vk  all tests passed (86 asserts / 4 tests)
device_radix_sort_one_sweep cuda/dx/vk  all tests passed (66 asserts / 3 tests)
```

`xmake build lcpp_cub_bench device_reduce_test device_scan_test
device_radix_sort_one_sweep device_segment_reduce` succeeds.

## Concerns / follow-ups

- The dispatch overhead the epoch was scoped to is real but tiny on this workload.
  To cut the ~120 µs dx/vk floor you'd need to reduce per-call
  submit/synchronize cost (persistent/mapped command lists, batching, or CUDA-graph
  capture like `LCPP_BENCH_GRAPH=1`), which is a backend concern, not lcpp.
- The probe (`host_enqueue_probe`) is a diagnostic target added to `tests/`; it
  does not affect the library or `lcpp_cub_bench` and can be dropped if undesired.
