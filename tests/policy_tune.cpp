// Epoch D policy-tuning harness (measurement / evidence tool).
//
// The one-sweep radix tile (and the reduce tile) are compile-time template
// parameters, so a sweep needs several instantiations.  This harness
// instantiates a small fixed set of tiles once and then, at runtime, measures
// the requested subset on the **DSL** path and -- on vk, when the
// `LCPP_INTEROP` policy selects the module -- on the **interop** path, in one
// interleaved round-robin so slow drift affects every case equally.
//
// Every case is verified bit-exactly against a host reference before its
// timings are reported, so a tile that is fast but wrong cannot be adopted.
//
// usage: policy_tune <backend> [--op=radix|reduce|all] [--items=N] [--reps=R]
//        [--batch=B] [--tiles=<t>[,<t>...]] [--reduce-tiles=<t>[,<t>...]]
//
//   backend        cuda | dx | vk | ...
//   --items=N      number of elements (default 4,194,304)
//   --reps=R       interleaved rounds (default 5)
//   --batch=B      samples per round; B=1 exposes the host floor, B>=100
//                  amortizes it (default 100)
//   --cl-ops=K     K op invocations recorded into ONE CommandList per sample
//                  (the epoch-C C3 pattern; default 1)
//   --tiles=...    radix one-sweep tiles to sweep, from {8,16,24,32,40};
//                  default 16,24,32
//   --reduce-tiles reduce tiles from {4,8,16}; default 8,16
//   --dsl-max-items=N  largest tile instantiated on the DSL path (default 31:
//                  the vk/dx HLSL codegen enforces a 32 KiB shared-memory
//                  floor, which for a keys-only one-sweep tile of 256 threads
//                  is reached at 31 items/thread).  Set 48 for a pure-CUDA run.
//
// Env: LCPP_INTEROP selects the interop modules (vk only).  Run with
// `LCPP_INTEROP=radix,reduce` (or `auto`) to get the interop rows.
//
// Output rows (CSV, stdout):
//   PT,<op>,<tile>,<path>,<items>,<median_ms>,<min_ms>,<max_ms>,<rounds>,<ok>
// `ok` is 1 only when the host-exactness check passed for every round's inputs.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <lcpp/common/utils.h>
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

constexpr size_t kBlock = 256u;
constexpr size_t kWarp  = 32u;

struct Case
{
    std::string                       op;
    size_t                            tile = 0;
    std::string                       path;
    std::function<void(CommandList&)> record;
    std::function<bool()>             verify;
    std::vector<float>                values;
    bool                              exact = false;
};

/// One timed sample: `batch` commits, each holding `cl_ops` op invocations in a
/// single CommandList, bracketed by one synchronize pair; per-op wall time.
/// `cl_ops > 1` is the epoch-C C3 pattern: it amortizes the per-commit host cost
/// (one submit + one completion round-trip per CommandList) over K ops, which is
/// what exposes the GPU-side difference of two tiles on the vk pipeline.
float measure(Stream&                       stream,
              int                           batch,
              int                           cl_ops,
              const std::function<void(CommandList&)>& record) noexcept
{
    stream.synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < batch; ++i)
    {
        CommandList cl;
        for(int k = 0; k < cl_ops; ++k)
        {
            record(cl);
        }
        stream << cl.commit();
    }
    stream.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count() /
           static_cast<float>(batch * cl_ops);
}

[[nodiscard]] float median_of(std::vector<float> v)
{
    if(v.empty())
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2u];
}

/// Interleaved round-robin measurement of every case (all cases measured once
/// per round, in order).
void sweep(std::vector<Case>& cases, Stream& stream, int reps, int batch, int cl_ops)
{
    for(auto& c : cases)
    {
        c.values.reserve(static_cast<size_t>(reps));
    }
    // Warm-up: first use of a case lazily compiles / imports its kernels; that
    // must never be inside a timed sample.
    for(auto& c : cases)
    {
        (void)measure(stream, 1, cl_ops, c.record);
    }
    for(int r = 0; r < reps; ++r)
    {
        for(auto& c : cases)
        {
            c.values.emplace_back(measure(stream, batch, cl_ops, c.record));
        }
    }
    for(auto& c : cases)
    {
        c.exact = c.verify();
    }
}

void report(std::vector<Case>& cases, size_t items)
{
    for(auto& c : cases)
    {
        const auto med = median_of(c.values);
        const auto mn  = *std::min_element(c.values.begin(), c.values.end());
        const auto mx  = *std::max_element(c.values.begin(), c.values.end());
        std::printf("PT,%s,%zu,%s,%zu,%.6f,%.6f,%.6f,%zu,%d\n",
                    c.op.c_str(),
                    c.tile,
                    c.path.c_str(),
                    items,
                    med,
                    mn,
                    mx,
                    c.values.size(),
                    c.exact ? 1 : 0);
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// DeviceRadixSort::SortKeys tile sweep
// ---------------------------------------------------------------------------
template <size_t TILE>
void add_radix_cases(std::vector<Case>&          cases,
                     Device&                     device,
                     Stream&                     stream,
                     InteropContext*             interop,
                     size_t                      n,
                     const std::vector<uint32_t>& host_in,
                     const std::vector<uint32_t>& host_ref,
                     bool                        with_interop,
                     bool                        with_dsl)
{
    using SortT = DeviceRadixSort<kBlock, kWarp, TILE>;
    auto keys_in  = std::make_shared<Buffer<uint32_t>>(device.create_buffer<uint32_t>(n));
    auto keys_out = std::make_shared<Buffer<uint32_t>>(device.create_buffer<uint32_t>(n));
    auto temp     = std::make_shared<Buffer<uint>>(
        device.create_buffer<uint>(
            bytes_to_uint_count(SortT::template GetSortKeysTempStorageBytes<uint32_t>(static_cast<uint>(n)))));
    stream << keys_in->copy_from(luisa::span{host_in}) << synchronize();

    auto make_case = [&](const char* path, InteropContext* ctx) -> Case
    {
        auto sorter = std::make_shared<SortT>();
        sorter->create(device, &stream, ctx);
        Case c;
        c.op   = "radix";
        c.tile = TILE;
        c.path = sorter->interop_active() ? "interop" : "dsl";
        if(ctx != nullptr && !sorter->interop_active())
        {
            std::fprintf(stderr, "[warn] radix tile %zu: interop context passed but the module "
                                 "stayed on the DSL path (policy did not select radix?)\n",
                         TILE);
        }
        c.record = [sorter, keys_in, keys_out, temp, n](CommandList& cl) noexcept
        { sorter->SortKeys(cl, temp->view(), keys_in->view(), keys_out->view(), static_cast<uint>(n)); };
        c.verify = [&stream, keys_out, host_ref, n]() noexcept
        {
            std::vector<uint32_t> result(n);
            stream << keys_out->copy_to(luisa::span{result}) << synchronize();
            return result == host_ref;
        };
        return c;
    };

    if(with_dsl)
    {
        cases.emplace_back(make_case("dsl", nullptr));
    }
    if(with_interop && interop != nullptr)
    {
        cases.emplace_back(make_case("interop", interop));
    }
}

// ---------------------------------------------------------------------------
// DeviceReduce::Sum tile sweep
// ---------------------------------------------------------------------------
template <size_t TILE>
void add_reduce_cases(std::vector<Case>&         cases,
                      Device&                    device,
                      Stream&                    stream,
                      InteropContext*            interop,
                      size_t                     n,
                      const std::vector<int32_t>& host_in,
                      int32_t                    expected,
                      bool                       with_interop,
                      bool                       with_dsl)
{
    using RedT = DeviceReduce<kBlock, kWarp, TILE>;
    auto in   = std::make_shared<Buffer<int32_t>>(device.create_buffer<int32_t>(n));
    auto out  = std::make_shared<Buffer<int32_t>>(device.create_buffer<int32_t>(1u));
    auto temp = std::make_shared<Buffer<uint>>(
        device.create_buffer<uint>(
            bytes_to_uint_count(RedT::template GetTempStorageBytes<int32_t>(n))));
    stream << in->copy_from(luisa::span{host_in}) << synchronize();

    auto make_case = [&](const char* path, InteropContext* ctx) -> Case
    {
        auto reducer = std::make_shared<RedT>();
        reducer->create(device, &stream, ctx);
        Case c;
        c.op   = "reduce";
        c.tile = TILE;
        c.path = reducer->interop_active() ? "interop" : "dsl";
        c.record = [reducer, in, out, temp, n](CommandList& cl) noexcept
        { reducer->Sum(cl, temp->view(), in->view(), out->view(), n); };
        c.verify = [&stream, out, expected]() noexcept
        {
            int32_t result = 0;
            stream << out->copy_to(luisa::span{&result, 1u}) << synchronize();
            return result == expected;
        };
        return c;
    };

    if(with_dsl)
    {
        cases.emplace_back(make_case("dsl", nullptr));
    }
    if(with_interop && interop != nullptr)
    {
        cases.emplace_back(make_case("interop", interop));
    }
}

[[nodiscard]] std::vector<size_t> parse_tiles(std::string_view text)
{
    std::vector<size_t> tiles;
    while(!text.empty())
    {
        auto comma = text.find(',');
        auto token = text.substr(0u, comma);
        if(!token.empty())
        {
            tiles.emplace_back(static_cast<size_t>(std::strtoull(std::string(token).c_str(), nullptr, 10)));
        }
        if(comma == std::string_view::npos)
        {
            break;
        }
        text.remove_prefix(comma + 1u);
    }
    return tiles;
}

[[nodiscard]] bool contains(const std::vector<size_t>& v, size_t x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string       backend = "vk";
    std::string       op      = "all";
    size_t            items   = 4u << 20u;
    int               reps    = 5;
    int               batch   = 100;
    int               cl_ops  = 1;
    std::vector<size_t> radix_tiles{16u, 24u, 32u};
    std::vector<size_t> reduce_tiles{8u, 16u};
    size_t              dsl_max_items = 31u;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg(argv[i]);
        if(arg.starts_with("--op="))
        {
            op = std::string(arg.substr(5u));
        }
        else if(arg.starts_with("--items="))
        {
            items = std::strtoull(std::string(arg.substr(8u)).c_str(), nullptr, 10);
        }
        else if(arg.starts_with("--reps="))
        {
            reps = std::atoi(std::string(arg.substr(7u)).c_str());
        }
        else if(arg.starts_with("--batch="))
        {
            batch = std::atoi(std::string(arg.substr(8u)).c_str());
        }
        else if(arg.starts_with("--cl-ops="))
        {
            cl_ops = std::atoi(std::string(arg.substr(9u)).c_str());
        }
        else if(arg.starts_with("--tiles="))
        {
            radix_tiles = parse_tiles(arg.substr(8u));
        }
        else if(arg.starts_with("--reduce-tiles="))
        {
            reduce_tiles = parse_tiles(arg.substr(15u));
        }
        else if(arg.starts_with("--dsl-max-items="))
        {
            dsl_max_items = std::strtoull(std::string(arg.substr(16u)).c_str(), nullptr, 10);
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
    reps   = std::max(1, reps);
    batch  = std::max(1, batch);
    cl_ops = std::max(1, cl_ops);

    Context context{argv[0]};
    Device  device = context.create_device(backend.c_str());
    if(!device)
    {
        std::fprintf(stderr, "failed to create the %s device\n", backend.c_str());
        return 2;
    }
    Stream stream = device.create_stream();

    // The interop context is vk-only and feature-detected; when unavailable the
    // harness silently measures the DSL path only.
    luisa::unique_ptr<InteropContext> interop;
    if(backend == "vk")
    {
        interop = InteropContext::create(context, device);
    }
    const bool with_interop = interop != nullptr;

    std::vector<Case> cases;

    if(op == "all" || op == "radix")
    {
        std::vector<uint32_t> host_in(items);
        for(size_t i = 0; i < items; ++i)
        {
            host_in[i] = static_cast<uint32_t>((i * 2654435761ull) ^ (i >> 7u));
        }
        auto host_ref = host_in;
        std::sort(host_ref.begin(), host_ref.end());
        auto add = [&](size_t tile)
        {
            switch(tile)
            {
                case 8u: add_radix_cases<8u>(cases, device, stream, interop.get(), items, host_in, host_ref, with_interop, 8u <= dsl_max_items); break;
                case 16u: add_radix_cases<16u>(cases, device, stream, interop.get(), items, host_in, host_ref, with_interop, 16u <= dsl_max_items); break;
                case 24u: add_radix_cases<24u>(cases, device, stream, interop.get(), items, host_in, host_ref, with_interop, 24u <= dsl_max_items); break;
                case 32u: add_radix_cases<32u>(cases, device, stream, interop.get(), items, host_in, host_ref, with_interop, 32u <= dsl_max_items); break;
                case 40u: add_radix_cases<40u>(cases, device, stream, interop.get(), items, host_in, host_ref, with_interop, 40u <= dsl_max_items); break;
                default: std::fprintf(stderr, "[warn] radix tile %zu is not instantiated (use 8/16/24/32/40)\n", tile); break;
            }
        };
        for(auto t : radix_tiles)
        {
            add(t);
        }
    }

    if(op == "all" || op == "reduce")
    {
        std::vector<int32_t> host_in(items);
        int32_t              expected = 0;
        for(size_t i = 0; i < items; ++i)
        {
            host_in[i] = static_cast<int32_t>(i % 13u) - 6;
            expected   = static_cast<int32_t>(static_cast<uint32_t>(expected) + static_cast<uint32_t>(host_in[i]));
        }
        auto add = [&](size_t tile)
        {
            switch(tile)
            {
                case 4u: add_reduce_cases<4u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 4u <= dsl_max_items); break;
                case 8u: add_reduce_cases<8u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 8u <= dsl_max_items); break;
                case 16u: add_reduce_cases<16u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 16u <= dsl_max_items); break;
                case 24u: add_reduce_cases<24u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 24u <= dsl_max_items); break;
                case 32u: add_reduce_cases<32u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 32u <= dsl_max_items); break;
                case 40u: add_reduce_cases<40u>(cases, device, stream, interop.get(), items, host_in, expected, with_interop, 40u <= dsl_max_items); break;
                default: std::fprintf(stderr, "[warn] reduce tile %zu is not instantiated (use 4/8/16)\n", tile); break;
            }
        };
        for(auto t : reduce_tiles)
        {
            add(t);
        }
    }

    if(cases.empty())
    {
        std::fprintf(stderr, "no cases selected\n");
        return 2;
    }

    std::printf("# backend=%s op=%s items=%zu reps=%d batch=%d interop=%s\n",
                backend.c_str(),
                op.c_str(),
                items,
                reps,
                batch,
                with_interop ? "available" : "unavailable");
    std::fflush(stdout);

    sweep(cases, stream, reps, batch, cl_ops);
    report(cases, items);

    auto failures = size_t{0};
    for(auto& c : cases)
    {
        if(!c.exact)
        {
            std::fprintf(stderr, "[FAIL] %s tile=%zu path=%s: host-exactness check failed\n",
                         c.op.c_str(),
                         c.tile,
                         c.path.c_str());
            ++failures;
        }
    }
    if(failures != 0u)
    {
        std::fprintf(stderr, "%zu/%zu cases failed exactness\n", failures, cases.size());
        return 1;
    }
    return 0;
}
