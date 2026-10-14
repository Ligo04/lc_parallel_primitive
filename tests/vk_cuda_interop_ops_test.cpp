// Epoch B (vk-cuda interop) exactness test for the *full* lcpp op surface.
//
// Every op is run twice on the same vk device:
//   * interop  -- the kernel is compiled by the cuda backend, imported through
//                 VkCudaInterop::create_cuda_kernel and launched by
//                 vkCmdCudaLaunchKernelNV inside the caller's command buffer;
//   * DSL      -- the pre-existing pure-vulkan SPIR-V path (regression
//                 reference, risk R8).
// Both results are compared bit-exactly against a host reference and against
// each other, at 4,194,304 items (the campaign acceptance size) and at the
// ragged 1,000,003 (multi-tile look-back, non-multiple of the tile).
//
// usage: vk_cuda_interop_ops_test [vk] [--op=scan|segment|arg|rbk|radix|all]
// Skips (exit 0) when the vk-cuda interop fast path is unavailable.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <lcpp/parallel_primitive.h>
#include <lcpp/runtime/interop_dispatch.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

namespace
{

int  g_failures = 0;
long g_checks   = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if(!ok)
    {
        std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
        ++g_failures;
    }
    else
    {
        std::printf("[ ok ] %s\n", what.c_str());
    }
}
std::string label(std::string_view op, size_t n, std::string_view what)
{
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%.*s(%zu).%.*s",
                  static_cast<int>(op.size()), op.data(), n,
                  static_cast<int>(what.size()), what.data());
    return buf;
}

constexpr size_t kBigItemsDefault        = 4u << 20u;  // 4,194,304
constexpr size_t kRaggedItemsDefault     = 1000003u;
constexpr size_t kSmallRaggedItemsDefault = 65537u;
size_t           g_big_items             = kBigItemsDefault;
size_t           g_ragged_items          = kRaggedItemsDefault;
size_t           g_small_ragged_items    = kSmallRaggedItemsDefault;
[[nodiscard]] std::vector<size_t> campaign_sizes()
{
    return {g_small_ragged_items, g_ragged_items, g_big_items};
}

/// Deterministic, full-range int32 input with plenty of duplicate keys.
std::vector<int32_t> make_input(size_t n)
{
    std::vector<int32_t> host(n);
    for(size_t i = 0; i < n; ++i)
    {
        host[i] = static_cast<int32_t>((i * 2654435761u) % 4099u) - 2049;
    }
    return host;
}

/// Distinct keys whenever the host index crosses a segment boundary.
std::vector<int32_t> make_segment_keys(size_t n, size_t segment_items)
{
    std::vector<int32_t> keys(n);
    for(size_t i = 0; i < n; ++i)
    {
        keys[i] = static_cast<int32_t>((i / segment_items) * 7u + (i % 3u));
    }
    return keys;
}

/// Module pairs: `interop` has the fast path enabled, `dsl` does not.
template <typename Module>
struct Pair
{
    Module          interop;
    Module          dsl;

    template <typename... Args>
    void create(Device& device, Stream& stream, InteropContext* interop_context, Args&&... args)
    {
        interop.create(device, &stream, interop_context, std::forward<Args>(args)...);
        dsl.create(device, &stream, nullptr, std::forward<Args>(args)...);
    }
};

// Epoch C: the LCPP_INTEROP policy (1|all|auto|a,b,c) selects the modules that
// take the fast path.  A module the policy did not select is *expected* to stay
// on the DSL path -- report it as skipped instead of failing (risk R3 is still
// enforced for every module the policy does select).
[[nodiscard]] bool require_active(bool active, const char* name, const char* policy_name)
{
    if(!active)
    {
        if(!interop_policy_selects(policy_name))
        {
            std::printf("[info] %s: not selected by the LCPP_INTEROP policy; DSL path (skipped)\n", name);
            return false;
        }
        std::fprintf(stderr, "[FAIL] %s: the interop module did not enable the fast path\n", name);
        ++g_failures;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// DeviceScan
// ---------------------------------------------------------------------------

using ScanT = DeviceScan<>;
// ScanByKey allocates *two* shared arrays of `shared_mem_size` elements, so the
// default 256x15 scan tile (30.9 KiB + block-scan temps) exceeds the 32 KiB
// HLSL/SPIR-V floor that also binds the pure-vulkan DSL reference path.  The
// by-key cases therefore use the policy the existing device_scan_test uses.
using ScanByKeyT = DeviceScan<256u, 32u, 4u>;

template <typename T>
void scan_case(Pair<ScanT>& modules,
               Device&     device,
               Stream&     stream,
               size_t      n,
               const char* op,
               bool        inclusive)
{
    auto host = make_input(n);

    auto in  = device.create_buffer<int32_t>(n);
    auto out = device.create_buffer<int32_t>(n);
    auto temp = device.create_buffer<uint>(
        bytes_to_uint_count(ScanT::GetTempStorageBytes<int32_t>(n)));
    stream << in.copy_from(luisa::span{host}) << synchronize();

    // host reference
    std::vector<int64_t> ref(n);
    int64_t              running = 0;
    for(size_t i = 0; i < n; ++i)
    {
        if(inclusive)
        {
            running += host[i];
            ref[i] = running;
        }
        else
        {
            ref[i] = running;
            running += host[i];
        }
    }

    auto run = [&](ScanT& scanner) noexcept
    {
        CommandList cmdlist;
        if(inclusive)
        {
            scanner.InclusiveSum(cmdlist, temp.view(), in.view(), out.view(), n);
        }
        else
        {
            scanner.ExclusiveSum(cmdlist, temp.view(), in.view(), out.view(), n);
        }
        stream << cmdlist.commit() << synchronize();
        std::vector<int32_t> result(n);
        stream << out.copy_to(luisa::span{result}) << synchronize();
        return result;
    };

    auto interop_result = run(modules.interop);
    auto dsl_result     = run(modules.dsl);

    auto mismatches      = size_t{0};
    auto dsl_mismatches  = size_t{0};
    for(size_t i = 0; i < n; ++i)
    {
        if(static_cast<int64_t>(interop_result[i]) != ref[i])
        {
            if(mismatches < 3u)
            {
                std::fprintf(stderr, "[FAIL] %s[%zu]: interop=%d want=%lld\n",
                             op, i, interop_result[i], static_cast<long long>(ref[i]));
            }
            ++mismatches;
        }
        if(interop_result[i] != dsl_result[i])
        {
            ++dsl_mismatches;
        }
    }
    check(mismatches == 0u, label(op, n, inclusive ? "InclusiveSum==host" : "ExclusiveSum==host"));
    check(dsl_mismatches == 0u, label(op, n, "interop==dsl"));
}

void test_scan(Device& device, Stream& stream, InteropContext* interop)
{
    Pair<ScanT> modules;
    modules.create(device, stream, interop);
    if(!require_active(modules.interop.interop_active(), "DeviceScan", "scan"))
    {
        return;
    }
    ScanT& scanner = modules.interop;
    for(auto n : campaign_sizes())
    {
        scan_case<ScanT>(modules, device, stream, n, "scan", false);
        scan_case<ScanT>(modules, device, stream, n, "scan", true);
    }
    std::printf("[info] DeviceScan interop_active=%d\n", scanner.interop_active() ? 1 : 0);

    // ScanByKey shares the module/maps but has its own two kernels.  It needs a
    // smaller tile (see ScanByKeyT) to fit the 32 KiB shared-memory floor.
    Pair<ScanByKeyT> by_key;
    by_key.create(device, stream, interop);
    if(!require_active(by_key.interop.interop_active(), "DeviceScan.ScanByKey", "scan"))
    {
        return;
    }
    for(auto n : campaign_sizes())
    {
        const size_t segment_items = 1000u;
        auto         keys          = make_segment_keys(n, segment_items);
        auto         values        = make_input(n);

        auto key_buf = device.create_buffer<int32_t>(n);
        auto val_in  = device.create_buffer<int32_t>(n);
        auto val_out = device.create_buffer<int32_t>(n);
        auto temp    = device.create_buffer<uint>(
            bytes_to_uint_count(ScanByKeyT::GetScanByKeyTempStorageBytes<int32_t, int32_t>(n)));
        stream << key_buf.copy_from(luisa::span{keys}) << val_in.copy_from(luisa::span{values})
               << synchronize();

        std::vector<int32_t> ref(n);
        int64_t              running = 0;
        for(size_t i = 0; i < n; ++i)
        {
            if(i == 0u || keys[i] != keys[i - 1u])
            {
                running = 0;
            }
            ref[i] = static_cast<int32_t>(running);
            running += values[i];
        }

        auto run = [&](ScanByKeyT& s) noexcept
        {
            CommandList cmdlist;
            s.ExclusiveSumByKey(cmdlist, temp.view(), key_buf.view(), val_in.view(), val_out.view(), n);
            stream << cmdlist.commit() << synchronize();
            std::vector<int32_t> result(n);
            stream << val_out.copy_to(luisa::span{result}) << synchronize();
            return result;
        };
        auto interop_result = run(by_key.interop);
        auto dsl_result     = run(by_key.dsl);
        auto mismatches     = size_t{0};
        auto dsl_mismatches = size_t{0};
        for(size_t i = 0; i < n; ++i)
        {
            if(interop_result[i] != ref[i])
            {
                if(mismatches < 3u)
                {
                    std::fprintf(stderr, "[FAIL] scan_by_key[%zu]: interop=%d want=%d\n",
                                 i, interop_result[i], ref[i]);
                }
                ++mismatches;
            }
            if(interop_result[i] != dsl_result[i])
            {
                ++dsl_mismatches;
            }
        }
        check(mismatches == 0u, label("scan_by_key", n, "ExclusiveSumByKey==host"));
        check(dsl_mismatches == 0u, label("scan_by_key", n, "interop==dsl"));
    }
}

// ---------------------------------------------------------------------------
// DeviceSegmentReduce (variable-size and fixed-size, incl. ArgMin/ArgMax)
// ---------------------------------------------------------------------------

using SegmentT = DeviceSegmentReduce<>;

/// Per-segment reduce over `values[begin[s], end[s])` with the op semantics of
/// ArgMaxOp / ArgMinOp (ties broken by the larger / smaller *global* index).
struct SegmentRef
{
    std::vector<int32_t> sum;
    std::vector<int32_t> min;
    std::vector<int32_t> max;
    std::vector<uint32_t> arg_index;   // index relative to segment begin
    std::vector<int32_t>  arg_value;
};

SegmentRef segment_reference(const std::vector<int32_t>& values,
                             const std::vector<size_t>&  begins,
                             const std::vector<size_t>&  ends,
                             bool                        arg_max)
{
    SegmentRef ref;
    ref.sum.resize(begins.size());
    ref.min.resize(begins.size());
    ref.max.resize(begins.size());
    ref.arg_index.resize(begins.size());
    ref.arg_value.resize(begins.size());
    for(size_t s = 0; s < begins.size(); ++s)
    {
        int64_t sum = 0;
        int32_t mn  = std::numeric_limits<int32_t>::max();
        int32_t mx  = std::numeric_limits<int32_t>::lowest();
        // Mirrors ArgMaxOp {b.value > a.value | (b.value == a.value & b.key > a.key)}
        // starting from the lcpp seed IVP{1, lowest} / IVP{1, max}.
        uint32_t best_key = 1u;
        int32_t  best_val = arg_max ? std::numeric_limits<int32_t>::lowest() :
                                      std::numeric_limits<int32_t>::max();
        for(size_t i = begins[s]; i < ends[s]; ++i)
        {
            auto v = values[i];
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            auto key = static_cast<uint32_t>(i);
            if(arg_max ? (v > best_val || (v == best_val && key > best_key)) :
                         (v < best_val || (v == best_val && key < best_key)))
            {
                best_val = v;
                best_key = key;
            }
        }
        ref.sum[s]       = static_cast<int32_t>(sum);
        ref.min[s]       = mn;
        ref.max[s]       = mx;
        ref.arg_value[s] = best_val;
        ref.arg_index[s] = static_cast<uint32_t>(static_cast<int64_t>(best_key) - static_cast<int64_t>(begins[s]));
    }
    return ref;
}

void segment_case(Pair<SegmentT>&            modules,
                  Device&                    device,
                  Stream&                    stream,
                  size_t                     n,
                  const std::vector<size_t>& begins_host,
                  const std::vector<size_t>& ends_host,
                  const char*                label_prefix)
{
    const auto num_segments = static_cast<uint>(begins_host.size());
    auto       values       = make_input(n);

    auto in     = device.create_buffer<int32_t>(n);
    auto out    = device.create_buffer<int32_t>(num_segments);
    auto idxout = device.create_buffer<uint>(num_segments);
    auto begin_offsets = device.create_buffer<uint>(num_segments);
    auto end_offsets   = device.create_buffer<uint>(num_segments);
    std::vector<uint32_t> begins32(num_segments), ends32(num_segments);
    for(size_t s = 0; s < num_segments; ++s)
    {
        begins32[s] = static_cast<uint32_t>(begins_host[s]);
        ends32[s]   = static_cast<uint32_t>(ends_host[s]);
    }
    auto arg_temp = device.create_buffer<uint>(
        bytes_to_uint_count(SegmentT::GetArgTempStorageBytes<int32_t>(n, num_segments)));
    auto dummy_temp = device.create_buffer<uint>(1u);

    stream << in.copy_from(luisa::span{values}) << begin_offsets.copy_from(luisa::span{begins32})
           << end_offsets.copy_from(luisa::span{ends32}) << synchronize();

    auto ref = segment_reference(values, begins_host, ends_host, true);

    auto run_int = [&](SegmentT& reducer, int which) noexcept
    {
        CommandList cmdlist;
        // NOTE: the public `Sum`/`Min`/`Max` helpers of DeviceSegmentReduce
        // pass the *overloaded* `compute::min` / `compute::max` as the reduce
        // op, which the compiler cannot use to deduce `ReduceOp`, so those two
        // helpers do not compile today (pre-existing, unrelated to interop).
        // Driving the same kernel through the generic `Reduce` with the named
        // MinOp/MaxOp functors exercises exactly the same shader.
        switch(which)
        {
            case 0:
                reducer.Reduce(cmdlist, dummy_temp.view(), in.view(), out.view(), num_segments,
                               begin_offsets.view(), end_offsets.view(), SumOp(), int32_t(0));
                break;
            case 1:
                reducer.Reduce(cmdlist, dummy_temp.view(), in.view(), out.view(), num_segments,
                               begin_offsets.view(), end_offsets.view(),
                               MinOp(), std::numeric_limits<int32_t>::max());
                break;
            default:
                reducer.Reduce(cmdlist, dummy_temp.view(), in.view(), out.view(), num_segments,
                               begin_offsets.view(), end_offsets.view(),
                               MaxOp(), std::numeric_limits<int32_t>::lowest());
                break;
        }
        stream << cmdlist.commit() << synchronize();
        std::vector<int32_t> result(num_segments);
        stream << out.copy_to(luisa::span{result}) << synchronize();
        return result;
    };
    auto run_arg = [&](SegmentT& reducer, bool arg_max) noexcept
    {
        CommandList cmdlist;
        if(arg_max)
        {
            reducer.ArgMax(cmdlist, arg_temp.view(), in.view(), out.view(), idxout.view(), num_segments, begin_offsets.view(), end_offsets.view());
        }
        else
        {
            reducer.Argmin(cmdlist, arg_temp.view(), in.view(), out.view(), idxout.view(), num_segments, begin_offsets.view(), end_offsets.view());
        }
        stream << cmdlist.commit() << synchronize();
        std::vector<int32_t> values_out(num_segments);
        std::vector<uint32_t> index_out(num_segments);
        stream << out.copy_to(luisa::span{values_out}) << idxout.copy_to(luisa::span{index_out})
               << synchronize();
        return std::pair{values_out, index_out};
    };

    auto compare = [&](const std::vector<int32_t>& got, const std::vector<int32_t>& want, const char* what)
    {
        auto bad = size_t{0};
        for(size_t s = 0; s < num_segments; ++s)
        {
            if(got[s] != want[s])
            {
                if(bad < 3u)
                {
                    std::fprintf(stderr, "[FAIL] %s(%zu).%s[%zu]: got=%d want=%d\n",
                                 label_prefix, n, what, s, got[s], want[s]);
                }
                ++bad;
            }
        }
        check(bad == 0u, label(label_prefix, n, what));
    };
    auto compare_u32 = [&](const std::vector<uint32_t>& got, const std::vector<uint32_t>& want, const char* what)
    {
        auto bad = size_t{0};
        for(size_t s = 0; s < num_segments; ++s)
        {
            if(got[s] != want[s])
            {
                if(bad < 3u)
                {
                    std::fprintf(stderr, "[FAIL] %s(%zu).%s[%zu]: got=%u want=%u\n",
                                 label_prefix, n, what, s, got[s], want[s]);
                }
                ++bad;
            }
        }
        check(bad == 0u, label(label_prefix, n, what));
    };

    auto i_sum = run_int(modules.interop, 0);
    auto d_sum = run_int(modules.dsl, 0);
    compare(i_sum, ref.sum, "Sum==host");
    check(i_sum == d_sum, label(label_prefix, n, "Sum interop==dsl"));
    auto i_min = run_int(modules.interop, 1);
    auto d_min = run_int(modules.dsl, 1);
    compare(i_min, ref.min, "Min==host");
    check(i_min == d_min, label(label_prefix, n, "Min interop==dsl"));
    auto i_max = run_int(modules.interop, 2);
    auto d_max = run_int(modules.dsl, 2);
    compare(i_max, ref.max, "Max==host");
    check(i_max == d_max, label(label_prefix, n, "Max interop==dsl"));

    auto [i_av, i_ai] = run_arg(modules.interop, true);
    auto [d_av, d_ai] = run_arg(modules.dsl, true);
    compare(i_av, ref.arg_value, "ArgMax.value==host");
    compare_u32(i_ai, ref.arg_index, "ArgMax.index==host");
    check(i_av == d_av && i_ai == d_ai, label(label_prefix, n, "ArgMax interop==dsl"));
}

void segment_fixed_case(Pair<SegmentT>& modules,
                        Device&         device,
                        Stream&         stream,
                        size_t          n,
                        uint32_t        segment_size)
{
    const auto n_segments = static_cast<uint>(n / segment_size);
    auto       values     = make_input(n);

    auto in     = device.create_buffer<int32_t>(n);
    auto out    = device.create_buffer<int32_t>(n_segments);
    auto idxout = device.create_buffer<uint>(n_segments);
    auto arg_temp = device.create_buffer<uint>(
        bytes_to_uint_count(SegmentT::GetArgTempStorageBytes<int32_t>(n, n_segments)));
    auto dummy_temp = device.create_buffer<uint>(1u);
    stream << in.copy_from(luisa::span{values}) << synchronize();

    std::vector<size_t> begins(n_segments), ends(n_segments);
    for(size_t s = 0; s < n_segments; ++s)
    {
        begins[s] = s * segment_size;
        ends[s]   = begins[s] + segment_size;
    }
    auto ref = segment_reference(values, begins, ends, true);

    auto run_int = [&](SegmentT& reducer) noexcept
    {
        CommandList cmdlist;
        // Same pre-existing `compute::min/max` deduction caveat as above.
        reducer.Reduce(cmdlist, dummy_temp.view(), in.view(), out.view(), n_segments, segment_size,
                       SumOp(), int32_t(0));
        stream << cmdlist.commit() << synchronize();
        std::vector<int32_t> result(n_segments);
        stream << out.copy_to(luisa::span{result}) << synchronize();
        return result;
    };
    auto run_arg = [&](SegmentT& reducer) noexcept
    {
        CommandList cmdlist;
        reducer.ArgMax(cmdlist, arg_temp.view(), in.view(), out.view(), idxout.view(), n_segments, segment_size);
        stream << cmdlist.commit() << synchronize();
        std::vector<int32_t> values_out(n_segments);
        std::vector<uint32_t> index_out(n_segments);
        stream << out.copy_to(luisa::span{values_out}) << idxout.copy_to(luisa::span{index_out})
               << synchronize();
        return std::pair{values_out, index_out};
    };

    char what[64];
    std::snprintf(what, sizeof(what), "fixed(%u).Sum==host", segment_size);
    auto i_sum = run_int(modules.interop);
    auto d_sum = run_int(modules.dsl);
    auto bad   = size_t{0};
    for(size_t s = 0; s < n_segments; ++s)
    {
        if(i_sum[s] != ref.sum[s])
        {
            if(bad == 0u)
            {
                std::fprintf(stderr, "[FAIL] %s[%zu]: got=%d want=%d\n", what, s, i_sum[s], ref.sum[s]);
            }
            ++bad;
        }
    }
    check(bad == 0u, label("segment", n, what));
    check(i_sum == d_sum, label("segment", n, "fixed.Sum interop==dsl"));

    std::snprintf(what, sizeof(what), "fixed(%u).ArgMax==host", segment_size);
    auto [i_av, i_ai] = run_arg(modules.interop);
    auto [d_av, d_ai] = run_arg(modules.dsl);
    bad               = 0;
    for(size_t s = 0; s < n_segments; ++s)
    {
        if(i_av[s] != ref.arg_value[s] || i_ai[s] != ref.arg_index[s])
        {
            if(bad == 0u)
            {
                std::fprintf(stderr, "[FAIL] %s[%zu]: value got=%d want=%d index got=%u want=%u\n",
                             what, s, i_av[s], ref.arg_value[s], i_ai[s], ref.arg_index[s]);
            }
            ++bad;
        }
    }
    check(bad == 0u, label("segment", n, what));
    check(i_av == d_av && i_ai == d_ai, label("segment", n, "fixed.ArgMax interop==dsl"));
}

void test_segment(Device& device, Stream& stream, InteropContext* interop)
{
    Pair<SegmentT> modules;
    modules.create(device, stream, interop);
    if(!require_active(modules.interop.interop_active(), "DeviceSegmentReduce", "segment"))
    {
        return;
    }
    std::printf("[info] DeviceSegmentReduce interop_active=%d\n",
                modules.interop.interop_active() ? 1 : 0);

    for(auto n : campaign_sizes())
    {
        // Variable-size segments: 1..613 items wide, so the tile-crossing
        // (ConsumeFullTileRange) and the single-partial-tile paths are both hit.
        std::vector<size_t> begins, ends;
        size_t              pos = 0;
        for(size_t s = 0; pos < n; ++s)
        {
            auto len = 1u + static_cast<size_t>((s * 37u) % 613u);
            if(pos + len > n)
            {
                len = n - pos;
            }
            begins.push_back(pos);
            ends.push_back(pos + len);
            pos += len;
        }
        segment_case(modules, device, stream, n, begins, ends, "segment_var");
    }

    // Fixed-size segments: one below the warp-tile threshold (warp path) and
    // one far above it (block path).
    for(auto segment_size : {32u, 1000u})
    {
        segment_fixed_case(modules, device, stream, g_big_items, segment_size);
        segment_fixed_case(modules, device, stream, g_ragged_items, segment_size);
    }
}

// ---------------------------------------------------------------------------
// DeviceReduce remainder: ArgMin/ArgMax (struct-element reduce) + ReduceByKey
// ---------------------------------------------------------------------------

using ReduceT = DeviceReduce<256u, 32u, 8u>;

using IVP = IndexValuePairT<int32_t>;
using FVP = KeyValuePair<int, int32_t>;

void test_arg_reduce(Device& device, Stream& stream, InteropContext* interop)
{
    Pair<ReduceT> modules;
    modules.create(device, stream, interop);
    if(!require_active(modules.interop.interop_active(), "DeviceReduce.ArgMinMax", "reduce"))
    {
        return;
    }
    std::printf("[info] DeviceReduce.ArgMinMax interop_active=%d\n",
                modules.interop.interop_active() ? 1 : 0);

    // ABI gate (risk R2/R4 for struct arguments): the host-side aggregate that
    // the interop launcher memcpy's into the kernel's 16-byte-aligned Params
    // slot must have exactly the size the *device* DSL struct type has.
    check(Type::of<IVP>()->size() == sizeof(IVP),
          "ABI: sizeof(IndexValuePairT<int32>) == device struct size");
    check(Type::of<FVP>()->size() == sizeof(FVP),
          "ABI: sizeof(KeyValuePair<int,int32>) == device struct size");

    for(auto n : campaign_sizes())
    {
        auto host = make_input(n);

        auto in     = device.create_buffer<int32_t>(n);
        auto out    = device.create_buffer<int32_t>(1u);
        auto idxout = device.create_buffer<uint>(1u);
        auto temp   = device.create_buffer<uint>(
            bytes_to_uint_count(ReduceT::GetArgTempStorageBytes<int32_t>(n)));
        stream << in.copy_from(luisa::span{host}) << synchronize();

        // ArgMinOp / ArgMaxOp define a *total* order on (value, key) pairs, so
        // the winner is order-independent: smallest value / largest value, ties
        // broken by smallest / largest index respectively.
        size_t    min_idx = 0, max_idx = 0;
        int32_t   min_val = host[0], max_val = host[0];
        for(size_t i = 0; i < n; ++i)
        {
            if(host[i] < min_val)
            {
                min_val = host[i];
                min_idx = i;
            }
            if(host[i] >= max_val)
            {
                max_val = host[i];
                max_idx = i;
            }
        }

        auto run = [&](ReduceT& reducer, bool arg_max) noexcept
        {
            CommandList cmdlist;
            if(arg_max)
            {
                reducer.ArgMax(cmdlist, temp.view(), in.view(), out.view(), idxout.view(), n);
            }
            else
            {
                reducer.ArgMin(cmdlist, temp.view(), in.view(), out.view(), idxout.view(), n);
            }
            stream << cmdlist.commit() << synchronize();
            int32_t  value = 0;
            uint32_t index = 0;
            stream << out.copy_to(luisa::span{&value, 1u}) << idxout.copy_to(luisa::span{&index, 1u})
                   << synchronize();
            return std::pair{value, index};
        };

        auto [i_min_v, i_min_i] = run(modules.interop, false);
        auto [d_min_v, d_min_i] = run(modules.dsl, false);
        check(i_min_v == min_val && i_min_i == min_idx,
              label("argmin", n, "value/index==host"));
        check(i_min_v == d_min_v && i_min_i == d_min_i, label("argmin", n, "interop==dsl"));

        auto [i_max_v, i_max_i] = run(modules.interop, true);
        auto [d_max_v, d_max_i] = run(modules.dsl, true);
        check(i_max_v == max_val && i_max_i == max_idx,
              label("argmax", n, "value/index==host"));
        check(i_max_v == d_max_v && i_max_i == d_max_i, label("argmax", n, "interop==dsl"));
    }
}

void test_reduce_by_key(Device& device, Stream& stream, InteropContext* interop)
{
    Pair<ReduceT> modules;
    modules.create(device, stream, interop);
    if(!require_active(modules.interop.interop_active(), "DeviceReduce.ReduceByKey", "reduce"))
    {
        return;
    }
    std::printf("[info] DeviceReduce.ReduceByKey interop_active=%d\n",
                modules.interop.interop_active() ? 1 : 0);

    for(auto n : campaign_sizes())
    {
        constexpr size_t kRunLength = 7u;
        auto             values     = make_input(n);
        std::vector<int32_t> keys(n);
        for(size_t i = 0; i < n; ++i)
        {
            keys[i] = static_cast<int32_t>((i / kRunLength) % 1000u);
        }
        // host reference
        std::vector<int32_t> ref_keys;
        std::vector<int32_t> ref_values;
        for(size_t i = 0; i < n;)
        {
            size_t j = i;
            int64_t acc = 0;
            while(j < n && keys[j] == keys[i])
            {
                acc += values[j];
                ++j;
            }
            ref_keys.push_back(keys[i]);
            ref_values.push_back(static_cast<int32_t>(acc));
            i = j;
        }

        auto keys_in    = device.create_buffer<int32_t>(n);
        auto values_in  = device.create_buffer<int32_t>(n);
        auto keys_out   = device.create_buffer<int32_t>(n);
        auto values_out = device.create_buffer<int32_t>(n);
        auto runs_out   = device.create_buffer<uint>(1u);
        auto temp       = device.create_buffer<uint>(bytes_to_uint_count(
            ReduceT::GetReduceByKeyTempStorageBytes<int32_t, int32_t>(n)));
        stream << keys_in.copy_from(luisa::span{keys}) << values_in.copy_from(luisa::span{values})
               << synchronize();

        struct RbkResult
        {
            uint32_t             num_runs = 0;
            std::vector<int32_t> keys;
            std::vector<int32_t> values;
        };
        auto run = [&](ReduceT& reducer) noexcept
        {
            CommandList cmdlist;
            reducer.ReduceByKey(cmdlist,
                                temp.view(),
                                keys_in.view(),
                                values_in.view(),
                                keys_out.view(),
                                values_out.view(),
                                runs_out.view(),
                                SumOp(),
                                n);
            stream << cmdlist.commit() << synchronize();
            RbkResult result;
            // Read the whole output buffers back (a properly sized D2H) and
            // truncate on the host; `num_runs` validly bounds the prefix.
            std::vector<int32_t> keys_all(n), values_all(n);
            stream << runs_out.copy_to(luisa::span{&result.num_runs, 1u})
                   << keys_out.copy_to(luisa::span{keys_all})
                   << values_out.copy_to(luisa::span{values_all}) << synchronize();
            if(result.num_runs <= n)
            {
                result.keys.assign(keys_all.begin(), keys_all.begin() + result.num_runs);
                result.values.assign(values_all.begin(), values_all.begin() + result.num_runs);
            }
            return result;
        };
        auto dsl_result = run(modules.dsl);
        auto interop_result = run(modules.interop);

        auto ok = interop_result.num_runs == ref_keys.size()
                  && interop_result.keys == ref_keys && interop_result.values == ref_values;
        if(!ok && interop_result.num_runs != ref_keys.size())
        {
            std::fprintf(stderr, "[FAIL] rbk num_runs: got=%u want=%zu\n",
                         interop_result.num_runs, ref_keys.size());
        }
        check(ok, label("reduce_by_key", n, "keys/values/num_runs==host"));
        check(interop_result.num_runs == dsl_result.num_runs && interop_result.keys == dsl_result.keys
                  && interop_result.values == dsl_result.values,
              label("reduce_by_key", n, "interop==dsl"));
    }
}

// ---------------------------------------------------------------------------
// Diagnostic: D2H readback pattern probe (vk backend, interop-independent)
// ---------------------------------------------------------------------------
// A 4-byte D2H readback into a scalar followed by a second D2H readback of a
// larger span, each in its own commit(), was observed to segfault the host on
// the *pure-vulkan DSL* path (no interop launch involved) while developing the
// ReduceByKey test.  This reproduces that exact sequence in isolation so a
// future fix can be regression-tested; it does not exercise interop at all.
void test_d2h_probe(Device& device, Stream& stream, const std::string& mode)
{
    std::printf("[info] d2h-probe mode=%s\n", mode.c_str());
    if(mode != "partial-only" && mode != "full-only")
    {
        auto small = device.create_buffer<uint>(1u);
        std::vector<uint> zero{0u};
        stream << small.copy_from(luisa::span{zero}) << synchronize();
        uint value = 123u;
        stream << small.copy_to(luisa::span{&value, 1u}) << synchronize();
        check(value == 0u, "d2h-probe: 4-byte readback into a scalar");
    }

    constexpr size_t kSize = 65536u;
    auto             big  = device.create_buffer<int32_t>(kSize);
    std::vector<int32_t> host(kSize, 5);
    stream << big.copy_from(luisa::span{host}) << synchronize();
    if(mode == "full-only")
    {
        std::vector<int32_t> whole(kSize);
        stream << big.copy_to(luisa::span{whole}) << synchronize();
        check(whole.front() == 5 && whole.back() == 5, "d2h-probe: full readback");
        return;
    }
    if(mode == "aligned-partial")
    {
        // Partial *and* 16-byte-aligned (65520 * 4 = 262080 = 16 * 16380).
        std::vector<int32_t> part(kSize - 16u);
        stream << big.copy_to(luisa::span{part}) << synchronize();
        check(part.front() == 5 && part.back() == 5, "d2h-probe: 16B-aligned partial readback");
        return;
    }
    if(mode == "unaligned-full")
    {
        // Full readback of a 9363-element buffer (37452 bytes, not 16-aligned).
        auto small_buf = device.create_buffer<int32_t>(9363u);
        std::vector<int32_t> src(9363u, 7), dst(9363u);
        stream << small_buf.copy_from(luisa::span{src}) << synchronize();
        stream << small_buf.copy_to(luisa::span{dst}) << synchronize();
        check(dst.front() == 7 && dst.back() == 7, "d2h-probe: unaligned full readback");
        return;
    }
    std::vector<int32_t> partial(9363u);
    stream << big.copy_to(luisa::span{partial}) << synchronize();
    check(partial.front() == 5 && partial.back() == 5, "d2h-probe: partial readback");
}

// ---------------------------------------------------------------------------
// DeviceRadixSort (the campaign prize: 4 init + 2*passes launches)
// ---------------------------------------------------------------------------

using RadixT = DeviceRadixSort<>;

void test_radix(Device& device, Stream& stream, InteropContext* interop)
{
    Pair<RadixT> modules;
    modules.create(device, stream, interop);
    if(!require_active(modules.interop.interop_active(), "DeviceRadixSort", "radix"))
    {
        return;
    }
    std::printf("[info] DeviceRadixSort interop_active=%d\n",
                modules.interop.interop_active() ? 1 : 0);

    for(auto n : campaign_sizes())
    {
        auto keys_host = make_input(n);
        for(auto& k : keys_host)
        {
            k = static_cast<int32_t>(static_cast<uint32_t>(k) * 2654435761u);
        }
        std::vector<uint32_t> keys_u(n);
        for(size_t i = 0; i < n; ++i)
        {
            keys_u[i] = static_cast<uint32_t>(keys_host[i]);
        }
        auto ref = keys_u;
        std::sort(ref.begin(), ref.end());

        auto keys_in  = device.create_buffer<uint32_t>(n);
        auto keys_out = device.create_buffer<uint32_t>(n);
        auto temp = device.create_buffer<uint>(
            bytes_to_uint_count(RadixT::GetSortKeysTempStorageBytes<uint32_t>(static_cast<uint>(n))));
        stream << keys_in.copy_from(luisa::span{keys_u}) << synchronize();

        auto run = [&](RadixT& sorter) noexcept
        {
            CommandList cmdlist;
            sorter.SortKeys(cmdlist, temp.view(), keys_in.view(), keys_out.view(), static_cast<uint>(n));
            stream << cmdlist.commit() << synchronize();
            std::vector<uint32_t> result(n);
            stream << keys_out.copy_to(luisa::span{result}) << synchronize();
            return result;
        };
        auto interop_result = run(modules.interop);
        auto dsl_result     = run(modules.dsl);

        auto bad = size_t{0};
        for(size_t i = 0; i < n; ++i)
        {
            if(interop_result[i] != ref[i])
            {
                if(bad < 3u)
                {
                    std::fprintf(stderr, "[FAIL] radix[%zu]: got=%u want=%u\n",
                                 i, interop_result[i], ref[i]);
                }
                ++bad;
            }
        }
        check(bad == 0u, label("radix_sort", n, "SortKeys==host(sorted)"));
        check(interop_result == dsl_result, label("radix_sort", n, "interop==dsl"));
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string backend = "vk";
    std::string op      = "all";
    std::string d2h_mode = "default";
    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg(argv[i]);
        if(arg.starts_with("--op="))
        {
            op = std::string(arg.substr(5));
        }
        else if(arg.starts_with("--big="))
        {
            g_big_items = std::strtoull(std::string(arg.substr(6)).c_str(), nullptr, 10);
        }
        else if(arg.starts_with("--ragged="))
        {
            g_ragged_items = std::strtoull(std::string(arg.substr(9)).c_str(), nullptr, 10);
        }
        else if(arg.starts_with("--d2h-mode="))
        {
            d2h_mode = std::string(arg.substr(11));
        }
        else if(arg.starts_with("--"))
        {
            continue;
        }
        else
        {
            backend = std::string(arg);
        }
    }
    if(backend != "vk")
    {
        std::printf("vk_cuda_interop_ops_test requires the vk backend; skipping.\n");
        return 0;
    }

    log_level_verbose();
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep progress visible if we crash
    Context context{argv[0]};
    Device  device = context.create_device("vk");
    if(!device)
    {
        std::fprintf(stderr, "failed to create the vk device\n");
        return 2;
    }
    Stream stream = device.create_stream();

    // LCPP_INTEROP=0 disables the fast path entirely (useful to attribute a
    // behaviour to interop or to the host/vk path); the d2h diagnostic below is
    // interop-independent and runs either way.
    auto interop_requested = []
    {
        const char* env = std::getenv("LCPP_INTEROP");
        return env == nullptr || (env[0] != '\0' && env[0] != '0');
    }();
    luisa::unique_ptr<InteropContext> interop;
    if(interop_requested)
    {
        interop = InteropContext::create(context, device);
    }
    if(interop == nullptr)
    {
        if(op == "d2h")
        {
            std::printf("-- D2H readback probe (diagnostic, interop disabled) --\n");
            test_d2h_probe(device, stream, d2h_mode);
            return g_failures == 0 ? 0 : 1;
        }
        std::printf("vk-cuda interop is unavailable (or LCPP_INTEROP=0); skipping.\n");
        return 0;
    }
    std::printf("vk-cuda interop active: cuda_device_index=%d op=%s\n",
                interop->ext()->cuda_device_index(), op.c_str());

    auto want = [&](const char* name) noexcept
    { return op == "all" || op == name; };

    if(want("scan"))
    {
        std::printf("-- DeviceScan --\n");
        test_scan(device, stream, interop.get());
    }
    if(want("segment"))
    {
        std::printf("-- DeviceSegmentReduce --\n");
        test_segment(device, stream, interop.get());
    }
    if(want("arg"))
    {
        std::printf("-- DeviceReduce ArgMin/ArgMax --\n");
        test_arg_reduce(device, stream, interop.get());
    }
    if(op == "d2h")
    {
        std::printf("-- D2H readback probe (diagnostic, never part of --op=all) --\n");
        test_d2h_probe(device, stream, d2h_mode);
    }
    if(want("rbk"))
    {
        std::printf("-- DeviceReduce ReduceByKey --\n");
        test_reduce_by_key(device, stream, interop.get());
    }
    if(want("radix"))
    {
        std::printf("-- DeviceRadixSort --\n");
        test_radix(device, stream, interop.get());
    }

    if(g_failures == 0)
    {
        std::printf("PASS: all vk-cuda interop op checks are bit-exact (%ld checks).\n", g_checks);
    }
    else
    {
        std::printf("FAIL: %d of %ld checks failed.\n", g_failures, g_checks);
    }
    return g_failures == 0 ? 0 : 1;
}
