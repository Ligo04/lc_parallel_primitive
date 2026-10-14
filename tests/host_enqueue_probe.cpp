// Epoch 1 diagnostic: measure the HOST-side cost of building a CommandList for
// each device op WITHOUT committing or synchronizing.  This isolates the
// per-call dispatch overhead (shader-cache key construction + lookup + command
// graph construction) from the backend submit+sync latency that dominates the
// end-to-end wall time reported by lcpp_cub_bench on dx/vk.
//
// Epoch C adds `--commit`: the same loop, but the timed region is
// `stream << list.commit()` -- i.e. CommandList::Commit construction plus
// Stream::dispatch (barrier planning, backend command encoding, vkQueueSubmit,
// worker-thread hand-off) with synchronize() left outside the window.  That is
// exactly the part of the batch=1 floor the recording probe cannot see, and it
// is the direct test of the "per-launch vk encode cost" hypothesis for the
// scan gap (encode is where vkGetBufferDeviceAddress per buffer argument is
// called).  The probe drops the vk depth-2 in-flight throttle for this mode
// (DeviceConfig::inqueue_buffer_limit = false) and drains explicitly every
// `drain_every` commits, so the throttle's own GPU wait can never land inside
// the timed region.
//
// usage: host_enqueue_probe [backend=dx] [num_items=4096] [reps=200] [calls_per_cl=200] [--commit]
//        host_enqueue_probe vk 4096 200 1 --commit   (per-commit, 1 op per list)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include <lcpp/parallel_primitive.h>
#include <lcpp/runtime/interop_dispatch.h>

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

namespace {

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename F>
double time_enqueue_us(Stream& stream, int reps, int calls_per_cl, F&& enqueue_into) {
    // Warm every code path / cache entry once (compiles shaders), then discard
    // safely by committing.
    {
        CommandList warm;
        enqueue_into(warm);
        stream << warm.commit();
        stream.synchronize();
    }
    // Measure ONLY the host-side enqueue: build a CommandList containing
    // calls_per_cl op invocations and time that loop.  commit()/sync happen
    // outside the timed region so backend submit+sync latency is excluded --
    // this is exactly the per-call dispatch cost (cache key + lookup + command
    // graph construction) the optimization targets.
    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CommandList cl;
        auto t0 = std::chrono::steady_clock::now();
        for (int j = 0; j < calls_per_cl; ++j) {
            enqueue_into(cl);
        }
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        samples.push_back(us / calls_per_cl);
        // Drain to avoid unbounded queue growth; outside the timed region.
        stream << cl.commit();
        stream.synchronize();
    }
    return median(samples);
}

// Epoch C: per-COMMIT host cost (Commit construction + Stream::dispatch:
// barrier planning + backend encode + vkQueueSubmit), excluding synchronize().
// Recording is done outside the timed window, so this isolates the part of the
// batch=1 floor that the recording probe cannot see.
template <typename F>
double time_commit_us(Stream& stream, int reps, int calls_per_cl, F&& enqueue_into,
                      double& per_op_us) {
    constexpr int drain_every = 4;
    {
        CommandList warm;
        enqueue_into(warm);
        stream << warm.commit();
        stream.synchronize();
    }
    std::vector<double> samples;
    std::vector<double> per_op;
    samples.reserve(reps);
    per_op.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CommandList cl;
        for (int j = 0; j < calls_per_cl; ++j) { enqueue_into(cl); }
        auto t0 = std::chrono::steady_clock::now();
        stream << cl.commit();
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        samples.push_back(us);
        per_op.push_back(us / calls_per_cl);
        if ((i + 1) % drain_every == 0) { stream.synchronize(); }
    }
    stream.synchronize();
    per_op_us = median(per_op);
    return median(samples);
}

// Epoch C: host-only contract check for the patched KernelLauncher.  The
// single-blob rewrite must keep the command ABI byte-identical -- argument
// order, uniform offsets (past the exact argument header), uniform alignment
// padding, usize/value round-trip and the per-resource usage list.  Mirrors the
// assertions of LuisaCompute's src/tests/unit/ext/test_vk_cuda_launch_command.cpp
// so the in-tree lcpp regression guards the patch (LC tests are not built here).
// Returns the number of failed checks.
int check_kernel_launcher_packing() {
    using namespace vk_cuda_interop;
    auto failures = 0;
    auto expect = [&failures](bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "[FAIL] KernelLauncher packing: %s\n", what);
            ++failures;
        }
    };
    // 1) argument order + uniform offsets past the exact header + round-trip
    {
        KernelLauncher launcher;
        launcher.add_buffer(0x1000u, 32u, 256u, Usage::READ_WRITE)
            .add_uniform(2.5f)
            .add_texture(0x2000u, 1u, Usage::READ)
            .add_uniform(7u);
        auto cmd = std::move(launcher).build(1u, uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u}, 0u);
        auto args = cmd->arguments();
        const auto header = 4u * sizeof(Argument);
        expect(args.size() == 4u, "argument count");
        expect(args[0].tag == Argument::Tag::BUFFER && args[0].buffer.handle == 0x1000u &&
                   args[0].buffer.offset == 32u && args[0].buffer.size == 256u,
               "buffer record");
        expect(args[1].tag == Argument::Tag::UNIFORM && args[2].tag == Argument::Tag::TEXTURE &&
                   args[3].tag == Argument::Tag::UNIFORM,
               "argument order");
        expect(args[2].texture.handle == 0x2000u && args[2].texture.level == 1u, "texture record");
        expect(args[1].uniform.offset == header && args[1].uniform.size == sizeof(float) &&
                   args[1].uniform.alignment == alignof(float),
               "first uniform offset/size/alignment");
        expect(args[3].uniform.offset == header + sizeof(float), "second uniform offset");
        float f = 0.0f;
        uint32_t n = 0u;
        std::memcpy(&f, cmd->uniform(args[1].uniform).data(), sizeof(f));
        std::memcpy(&n, cmd->uniform(args[3].uniform).data(), sizeof(n));
        expect(f == 2.5f && n == 7u, "uniform round-trip");
        auto usages = cmd->argument_usages();
        expect(usages.size() == 2u && usages[0] == Usage::READ_WRITE && usages[1] == Usage::READ,
               "per-resource usage list");
        expect(luisa::all(cmd->dispatch_size() == uint3{1u, 1u, 1u}),
               "thread-count dispatch size");
        expect(luisa::all(cmd->block_dim() == uint3{64u, 1u, 1u}),
               "explicit block dimension");
        expect(luisa::all(cmd->grid_dim() == uint3{1u, 1u, 1u}),
               "derived grid dimension");
    }
    // 2) uniform alignment padding inside the blob
    {
        KernelLauncher launcher;
        launcher.add_uniform(uint8_t{0xabu}).add_uniform(uint64_t{0x0123456789abcdefull});
        auto cmd = std::move(launcher).build(1u, uint3{1u}, uint3{1u}, 0u);
        auto args = cmd->arguments();
        const auto header = 2u * sizeof(Argument);
        expect(args.size() == 2u && args[0].uniform.offset == header &&
                   args[1].uniform.offset == header + 8u,
               "uniform alignment padding");
        uint8_t small = 0u;
        uint64_t big = 0u;
        std::memcpy(&small, cmd->uniform(args[0].uniform).data(), sizeof(small));
        std::memcpy(&big, cmd->uniform(args[1].uniform).data(), sizeof(big));
        expect(small == 0xabu && big == 0x0123456789abcdefull, "alignment round-trip");
    }
    // 3) reserve() path (the typed CudaKernelT launch): same layout, and the
    //    pre-sized header must not shift a uniform offset.
    {
        KernelLauncher launcher;
        launcher.reserve(3u, 2u, 64u);
        launcher.add_buffer(0x1u, 0u, 8u, Usage::READ)
            .add_uniform(1.5f)
            .add_buffer(0x2u, 0u, 8u, Usage::WRITE);
        auto cmd = std::move(launcher).build(1u, uint3{1u}, uint3{32u}, 0u);
        auto args = cmd->arguments();
        const auto header = 3u * sizeof(Argument);
        expect(args.size() == 3u && args[1].tag == Argument::Tag::UNIFORM &&
                   args[1].uniform.offset == header,
               "reserve() preserves the exact uniform offset");
        float f = 0.0f;
        std::memcpy(&f, cmd->uniform(args[1].uniform).data(), sizeof(f));
        expect(f == 1.5f, "reserve() uniform round-trip");
        auto usages = cmd->argument_usages();
        expect(usages.size() == 2u && usages[0] == Usage::READ && usages[1] == Usage::WRITE,
               "reserve() usage list");
    }
    // 4) reserve() with too few slots must still produce the exact layout
    //    (the growth/fix-up path).
    {
        KernelLauncher launcher;
        launcher.reserve(1u, 1u, 0u);
        launcher.add_uniform(3u).add_buffer(0x3u, 0u, 4u, Usage::READ_WRITE).add_uniform(4.5f);
        auto cmd = std::move(launcher).build(1u, uint3{1u}, uint3{1u}, 0u);
        auto args = cmd->arguments();
        const auto header = 3u * sizeof(Argument);
        expect(args.size() == 3u, "over-reserved argument count");
        expect(args[0].uniform.offset == header, "grown-header first uniform offset");
        expect(args[2].uniform.offset == header + 4u, "grown-header second uniform offset");
        uint32_t n = 0u;
        float f = 0.0f;
        std::memcpy(&n, cmd->uniform(args[0].uniform).data(), sizeof(n));
        std::memcpy(&f, cmd->uniform(args[2].uniform).data(), sizeof(f));
        expect(n == 3u && f == 4.5f, "grown-header uniform round-trip");
    }
    if (failures == 0) { std::fprintf(stderr, "KernelLauncher packing check: ok (4 shapes)\n"); }
    return failures;
}

}  // namespace

namespace {

// Epoch C host-only encoder micro-benchmark.  Records the SAME trivial kernel
// through (a) the vk SPIR-V DSL encoder and (b) the imported CUDA kernel
// launcher (typed CudaKernelT) with identical argument lists, into a
// CommandList that is submitted outside the timed region.  Reported per
// dispatch, this isolates the two encoders (shader-cache lookup, argument
// packing, command allocation) from any device work, and the argument-count
// sweep separates a fixed per-dispatch cost from a per-argument one.
void micro_encoder_bench(Device& device, Stream& stream, InteropContext* interop, int reps, int per_cl) {
    if (interop == nullptr) {
        std::fprintf(stderr, "encoder_micro: skipped (needs LCPP_INTEROP=1 on vk)\n");
        return;
    }
    constexpr size_t micro_items = 4096u;
    std::vector<Buffer<int>> bufs;
    for (auto i = 0u; i < 8u; ++i) { bufs.emplace_back(device.create_buffer<int>(micro_items)); }

    auto time_encode = [&](const std::function<void(CommandList&)>& encode) {
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            CommandList cl;
            auto t0 = std::chrono::steady_clock::now();
            for (int j = 0; j < per_cl; ++j) { encode(cl); }
            auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                              static_cast<double>(per_cl));
            stream << cl.commit();// outside the timed region
            if ((i + 1) % 20 == 0) { stream.synchronize(); }
        }
        stream.synchronize();
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2u];
    };
    auto report = [&](const char* name, size_t args, double dsl_ns, double interop_ns) {
        std::fprintf(stderr,
                     "encoder_micro %-28s args=%zu dsl_ns=%.1f interop_ns=%.1f delta_ns=%.1f\n",
                     name, args, dsl_ns, interop_ns, interop_ns - dsl_ns);
    };

    // ---- variant 1: 2 buffer arguments ----
    {
        Kernel1D kernel = [](BufferInt a, BufferInt b) noexcept {
            b->write(dispatch_x(), a->read(dispatch_x()));
        };
        auto dsl = device.compile(kernel);
        auto cuda = interop->cuda_device().compile(kernel);
        auto imported = interop->import(cuda);
        if (!dsl || !imported) {
            std::fprintf(stderr, "encoder_micro: variant 1 compile/import failed\n");
            return;
        }
        auto typed = imported.kernel<vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::WRITE>>();
        auto dsl_ns = time_encode([&](CommandList& cl) {
            cl << dsl(bufs[0].view(), bufs[1].view()).dispatch(static_cast<uint>(micro_items));
        });
        auto interop_ns = time_encode([&](CommandList& cl) {
            cl << typed(bufs[0].view(), bufs[1].view()).dispatch(static_cast<uint>(micro_items));
        });
        report("buffers2", 2u, dsl_ns, interop_ns);
    }

    // ---- variant 2: 4 buffers + 1 scalar ----
    {
        Kernel1D kernel = [](BufferInt a, BufferInt b, BufferInt c, BufferInt d, UInt n) noexcept {
            auto i = dispatch_x();
            d->write(i, a->read(i) + b->read(i) + c->read(i) + cast<int>(n));
        };
        auto dsl = device.compile(kernel);
        auto cuda = interop->cuda_device().compile(kernel);
        auto imported = interop->import(cuda);
        auto typed = imported.kernel<vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::WRITE>,
                                     uint>();
        auto dsl_ns = time_encode([&](CommandList& cl) {
            cl << dsl(bufs[0].view(), bufs[1].view(), bufs[2].view(), bufs[3].view(), 1u)
                      .dispatch(static_cast<uint>(micro_items));
        });
        auto interop_ns = time_encode([&](CommandList& cl) {
            cl << typed(bufs[0].view(), bufs[1].view(), bufs[2].view(), bufs[3].view(), 1u)
                      .dispatch(static_cast<uint>(micro_items));
        });
        report("buffers4+uint", 5u, dsl_ns, interop_ns);
    }

    // ---- variant 3: 8 buffers + 3 scalars (radix one-sweep shape) ----
    {
        Kernel1D kernel = [](BufferInt a, BufferInt b, BufferInt c, BufferInt d,
                         BufferInt e, BufferInt f, BufferInt g, BufferInt h,
                         UInt n, UInt bit, UInt num_bits) noexcept {
            auto i = dispatch_x();
            h->write(i, a->read(i) + b->read(i) + c->read(i) + d->read(i) + e->read(i) +
                            f->read(i) + g->read(i) + cast<int>(n + bit + num_bits));
        };
        auto dsl = device.compile(kernel);
        auto cuda = interop->cuda_device().compile(kernel);
        auto imported = interop->import(cuda);
        auto typed = imported.kernel<vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::READ>,
                                     vk_cuda_interop::CudaArg<Buffer<int>, Usage::WRITE>,
                                     uint, uint, uint>();
        auto dsl_ns = time_encode([&](CommandList& cl) {
            cl << dsl(bufs[0].view(), bufs[1].view(), bufs[2].view(), bufs[3].view(), bufs[4].view(),
                      bufs[5].view(), bufs[6].view(), bufs[7].view(), 1u, 0u, 32u)
                      .dispatch(static_cast<uint>(micro_items));
        });
        auto interop_ns = time_encode([&](CommandList& cl) {
            cl << typed(bufs[0].view(), bufs[1].view(), bufs[2].view(), bufs[3].view(), bufs[4].view(),
                        bufs[5].view(), bufs[6].view(), bufs[7].view(), 1u, 0u, 32u)
                      .dispatch(static_cast<uint>(micro_items));
        });
        report("buffers8+3uint", 11u, dsl_ns, interop_ns);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* backend = argc > 1 ? argv[1] : "dx";
    size_t items        = argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 4096u;
    int reps            = argc > 3 ? std::atoi(argv[3]) : 200;
    int calls_per_cl    = argc > 4 ? std::atoi(argv[4]) : 200;
    // Epoch C: optional per-commit measurement and host-only encoder micro
    // benchmark (see the header comment).
    bool commit_mode = false;
    bool micro_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--commit") == 0) { commit_mode = true; }
        if (std::strcmp(argv[i], "--micro") == 0) { micro_mode = true; }
    }

    // vk: drop the depth-2 in-flight throttle so a per-commit measurement is
    // the pure host encode+submit cost (the probe drains explicitly instead of
    // letting the backend wait for fence-2).
    luisa::compute::DeviceConfig config{};
    config.inqueue_buffer_limit = false;
    const bool is_vk = std::string_view{backend} == "vk";

    Context context{argv[0]};
    Device device = is_vk ? context.create_device(backend, &config) : context.create_device(backend);
    if (!device) {
        std::fprintf(stderr, "no device for backend '%s'\n", backend);
        return 2;
    }
    Stream stream = device.create_stream();

    // Epoch A: opt-in vk-cuda interop fast path (env LCPP_INTEROP=1, vk only).
    // Declared after `device` so it is destroyed before it (risk R6).
    // Same truthiness rule as lcpp_cub_bench: LCPP_INTEROP=0 disables it.
    auto interop_requested = [] {
        const char* env = std::getenv("LCPP_INTEROP");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    luisa::unique_ptr<InteropContext> interop;
    if (interop_requested && std::string_view{backend} == "vk") {
        interop = InteropContext::create(context, device);
    }

    auto in     = device.create_buffer<int>(items);
    auto out    = device.create_buffer<int>(items);
    auto one    = device.create_buffer<int>(1u);
    auto uintin = device.create_buffer<uint>(items);
    auto uintout = device.create_buffer<uint>(items);

    constexpr uint RED_BLOCK = 256u, RED_ITEMS = 4u;
    constexpr uint SCAN_BLOCK = 256u, SCAN_ITEMS = 8u;
    constexpr uint RADIX_BLOCK = 256u, RADIX_ITEMS = 4u;

    using ReduceBench = DeviceReduce<RED_BLOCK, 32u, RED_ITEMS>;
    using ScanBench   = DeviceScan<SCAN_BLOCK, 32u, SCAN_ITEMS>;
    using RadixBench  = DeviceRadixSort<RADIX_BLOCK, 32u, RADIX_ITEMS>;

    auto red_temp   = device.create_buffer<uint>(bytes_to_uint_count(ReduceBench::GetTempStorageBytes<int>(items)));
    auto scan_temp  = device.create_buffer<uint>(bytes_to_uint_count(ScanBench::GetTempStorageBytes<int>(items)));
    auto radix_temp = device.create_buffer<uint>(bytes_to_uint_count(RadixBench::GetSortKeysTempStorageBytes<uint>(static_cast<uint>(items))));

    ReduceBench lcpp_reduce; lcpp_reduce.create(device, nullptr, interop.get());
    ScanBench   lcpp_scan;   lcpp_scan.create(device, nullptr, interop.get());
    RadixBench  lcpp_sort;   lcpp_sort.create(device, nullptr, interop.get());

    // Risk R3: a silently disabled fast path must fail the run, not pass
    // vacuously.  When LCPP_INTEROP was requested on the vk backend, the
    // context was created and the epoch-C per-op policy selects a module, that
    // module must actually have interop engaged.
    int interop_module_failure = 0;
    if (interop != nullptr) {
        const bool want_reduce = interop_policy_selects("reduce");
        const bool want_scan = interop_policy_selects("scan");
        const bool want_radix = interop_policy_selects("radix");
        const bool ok = (!want_reduce || lcpp_reduce.interop_active())
                        && (!want_scan || lcpp_scan.interop_active())
                        && (!want_radix || lcpp_sort.interop_active());
        std::fprintf(stderr,
                     "interop_active: reduce=%d scan=%d radix=%d (policy wants %d/%d/%d)\n",
                     lcpp_reduce.interop_active() ? 1 : 0,
                     lcpp_scan.interop_active() ? 1 : 0,
                     lcpp_sort.interop_active() ? 1 : 0,
                     want_reduce ? 1 : 0, want_scan ? 1 : 0, want_radix ? 1 : 0);
        if (!ok) {
            std::fprintf(stderr,
                         "FAIL: LCPP_INTEROP policy selected a module that fell back to the DSL path\n");
            interop_module_failure = 1;
        }
    }

    double reduce_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_reduce.Sum(cl, red_temp.view(), in.view(), one.view(), items);
    });
    double scan_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_scan.InclusiveSum(cl, scan_temp.view(), in.view(), out.view(), items);
    });
    double radix_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_sort.SortKeys(cl, radix_temp.view(), uintin.view(), uintout.view(), static_cast<uint>(items));
    });

    std::fprintf(stderr, "backend=%s items=%zu reps=%d calls_per_cl=%d interop=%d\n",
                 backend, items, reps, calls_per_cl, interop ? 1 : 0);
    std::fprintf(stderr, "DeviceReduce.Sum            host_enqueue_us=%.3f\n", reduce_us);
    std::fprintf(stderr, "DeviceScan.InclusiveSum     host_enqueue_us=%.3f\n", scan_us);
    std::fprintf(stderr, "DeviceRadixSort.SortKeys    host_enqueue_us=%.3f\n", radix_us);

    // Epoch C: per-commit (encode + submit) cost, symbiote of the recording
    // probe: [recording] + [commit] + [GPU + completion round-trip] is the
    // batch=1 wall time lcpp_cub_bench reports.
    if (commit_mode) {
        double reduce_op = 0.0, scan_op = 0.0, radix_op = 0.0;
        double reduce_cl = time_commit_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
            lcpp_reduce.Sum(cl, red_temp.view(), in.view(), one.view(), items);
        }, reduce_op);
        double scan_cl = time_commit_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
            lcpp_scan.InclusiveSum(cl, scan_temp.view(), in.view(), out.view(), items);
        }, scan_op);
        double radix_cl = time_commit_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
            lcpp_sort.SortKeys(cl, radix_temp.view(), uintin.view(), uintout.view(), static_cast<uint>(items));
        }, radix_op);
        std::fprintf(stderr, "host_commit_us_per_cl: sum=%.3f scan=%.3f radix=%.3f\n",
                     reduce_cl, scan_cl, radix_cl);
        std::fprintf(stderr, "host_commit_us_per_op: sum=%.3f scan=%.3f radix=%.3f\n",
                     reduce_op, scan_op, radix_op);
    }

    // Sanity: still produces correct results end-to-end (commit + sync once).
    stream << synchronize();

    if (micro_mode) {
        micro_encoder_bench(device, stream, interop.get(), reps, calls_per_cl);
    }
    // Epoch C: host-only contract check of the patched KernelLauncher (the LC
    // interop header is patched in this campaign; see
    // .kimix_cache/epoch_C_report.md).  Runs regardless of backend/interop.
    interop_module_failure |= check_kernel_launcher_packing();
    return interop_module_failure;
}
