#!/bin/bash
# Regression runner: all lcpp tests x {cuda, dx, vk} plus the vk-cuda interop
# mode (Epoch B) and the Epoch C per-op interop policy.
#
# Three modes per test:
#   * legacy  -- backend cuda / dx / vk on the pure-vulkan SPIR-V DSL dispatch
#                path.  This is the reference implementation; it must stay
#                bit-exact and unchanged (risk R8).
#   * interop -- backend vk with LCPP_INTEROP=1: the kernels are compiled by
#                the cuda backend and launched inside the vk command buffer
#                through VK_NV_cuda_kernel_launch.  Every module that supports
#                the fast path must report interop_active()==1; a silently
#                disabled fast path fails the run (risk R3), so the mode can
#                never pass vacuously.  This is the mode that verifies the
#                scan/radix interop kernels.
#   * policy  -- vk with LCPP_INTEROP=auto, the Epoch C measured per-op
#                recommendation (reduce/segment/radix interop, scan DSL).  The
#                tests only require interop for the modules the policy selects,
#                so this mode proves both the policy wiring and that the
#                unselected module really dispatches on the DSL path while
#                staying bit-exact.
#
# Ragged sizes 1,000,003 and 65,537 are exercised on every op by
# vk_cuda_interop_ops_test (its default size set).
#
# usage: scripts/run_regression.sh [--interop-only] [--legacy-only] [--policy-only]

cd "$(dirname "$0")/.."
export PATH="$PWD/LuisaCompute/bin/release:$PATH"
BIN=build/windows/x64/release

TESTS="block_level_test warp_level_test decoupled_look_back device_reduce_test device_scan_test device_segment_reduce device_radix_sort_one_sweep"
# Tests whose modules take the optional InteropContext (the rest are block/warp
# level tests with no Device* module).
INTEROP_AWARE="device_reduce_test device_scan_test device_segment_reduce device_radix_sort_one_sweep"

MODE="all"
for arg in "$@"; do
  case "$arg" in
    --interop-only) MODE="interop" ;;
    --legacy-only)  MODE="legacy" ;;
    --policy-only)  MODE="policy" ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

fail=0
total=0

run() {
  local label="$1"; shift
  local out rc last
  out=$("$@" 2>&1)
  rc=$?
  last=$(printf '%s\n' "$out" | tail -1)
  total=$((total + 1))
  if [ "$rc" -ne 0 ] || printf '%s\n' "$out" | grep -qE "\[FAIL\]|^FAIL"; then
    echo "FAIL $label :: rc=$rc :: $last"
    fail=$((fail + 1))
  else
    echo "PASS $label :: $last"
  fi
}

case "$MODE" in all|legacy) run_legacy=1 ;; esac
case "$MODE" in all|interop) run_interop=1 ;; esac
case "$MODE" in all|policy) run_policy=1 ;; esac

if [ -n "$run_legacy" ]; then
  echo "== legacy: pure-vulkan DSL dispatch path (reference implementation) =="
  for t in $TESTS; do
    for b in cuda dx vk; do
      run "$t $b" "$BIN/$t.exe" --backend="$b"
    done
  done
fi

if [ -n "$run_interop" ]; then
  echo "== vk-interop: cuda-compiled kernels in the vk command buffer =="
  for t in $TESTS; do
    if printf ' %s ' "$INTEROP_AWARE" | grep -q " $t "; then
      run "$t vk-interop" env LCPP_INTEROP=1 "$BIN/$t.exe" --backend=vk
    else
      run "$t vk-interop" "$BIN/$t.exe" --backend=vk
    fi
  done
  # Dedicated interop gates: R1/R1b/R3 + per-op exactness at
  # 4,194,304 / 1,000,003 / 65,537 (both modules run interop vs DSL vs host).
  # Epoch D: this gate must run with the interop policy *on* -- without
  # LCPP_INTEROP the module stays on the DSL path and the gate skips itself, so
  # the row would pass vacuously.  It also covers the single-tile / multi-tile
  # boundary (2048 / 2049 / 4096 items), which is exactly where a per-path tile
  # difference would change the recursion structure.
  run "vk_cuda_interop_reduce_test vk-interop" \
      env LCPP_INTEROP=1 "$BIN/vk_cuda_interop_reduce_test.exe" vk
  run "vk_cuda_interop_ops_test vk" \
      "$BIN/vk_cuda_interop_ops_test.exe" vk --op=all
  run "vk_cuda_interop_ops_test vk rounded sizes" \
      "$BIN/vk_cuda_interop_ops_test.exe" vk --op=all --big=65536 --ragged=65537
  # R3: the probe asserts every module the policy selects really engaged the
  # fast path.
  run "host_enqueue_probe vk-interop (R3 active assertion)" \
      env LCPP_INTEROP=1 "$BIN/host_enqueue_probe.exe" vk
fi

if [ -n "$run_policy" ]; then
  echo "== vk policy: LCPP_INTEROP=auto (Epoch C measured per-op selection) =="
  for t in $INTEROP_AWARE; do
    run "$t vk-policy(auto)" env LCPP_INTEROP=auto "$BIN/$t.exe" --backend=vk
  done
  run "vk_cuda_interop_ops_test vk policy auto" \
      env LCPP_INTEROP=auto "$BIN/vk_cuda_interop_ops_test.exe" vk --op=all
  run "host_enqueue_probe vk-policy(auto) (R3 per-op assertion)" \
      env LCPP_INTEROP=auto "$BIN/host_enqueue_probe.exe" vk
  run "vk_cuda_interop_ops_test vk policy list (reduce,radix)" \
      env LCPP_INTEROP=reduce,radix "$BIN/vk_cuda_interop_ops_test.exe" vk --op=all
  # Epoch D: the reduce R1/R1b gate again under the shipped recommendation
  # (`auto` selects reduce), so the narrow-integral interop tile is exercised on
  # the 2048/2049/4096 single-tile boundary and at 4M/ragged in the default mode.
  run "vk_cuda_interop_reduce_test vk-policy(auto)" \
      env LCPP_INTEROP=auto "$BIN/vk_cuda_interop_reduce_test.exe" vk
fi

echo "== $((total - fail))/$total passed, $fail failed =="
exit $((fail == 0 ? 0 : 1))
