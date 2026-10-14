#include <lcpp/parallel_primitive.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

#include <luisa/runtime/context.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/stream.h>
#include <lcpp/runtime/interop_dispatch.h>

// Portable defaults: keep per-block shared memory under the 32 KiB floor of
// the HLSL/DX codegen path ((BLOCK*ITEMS + BLOCK*ITEMS/32) * 4B <= 32 KiB).
// Override via -DLCPP_SCAN_BLOCK/-DLCPP_SCAN_ITEMS for backend-specific tuning.
#ifndef LCPP_SCAN_BLOCK
#define LCPP_SCAN_BLOCK 256u
#endif
#ifndef LCPP_SCAN_ITEMS
#define LCPP_SCAN_ITEMS 15u
#endif
#ifndef LCPP_SCAN_PACKED_TILE_STATE
#define LCPP_SCAN_PACKED_TILE_STATE 1
#endif
#ifndef LCPP_REDUCE_BLOCK
#define LCPP_REDUCE_BLOCK 256u
#endif
#ifndef LCPP_REDUCE_ITEMS
#define LCPP_REDUCE_ITEMS 8u
#endif
#ifndef LCPP_REDUCE_WARP_REDUX
#define LCPP_REDUCE_WARP_REDUX false
#endif
#ifndef LCPP_RADIX_BLOCK
#define LCPP_RADIX_BLOCK 256u
#endif
#ifndef LCPP_RADIX_ITEMS
#define LCPP_RADIX_ITEMS 16u
#endif
#ifndef LCPP_RADIX_CUB_POLICY
#define LCPP_RADIX_CUB_POLICY true
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

extern "C" {
int cub_bench_version();
cudaError_t cub_reduce_query(const int *, int *, int, cudaStream_t, size_t *);
cudaError_t cub_reduce_run(void *, size_t, const int *, int *, int, cudaStream_t);
cudaError_t cub_scan_query(const int *, int *, int, cudaStream_t, size_t *);
cudaError_t cub_scan_run(void *, size_t, const int *, int *, int, cudaStream_t);
cudaError_t cub_radix_query(const unsigned int *, unsigned int *, int, int, int, cudaStream_t, size_t *);
cudaError_t cub_radix_run(void *, size_t, const unsigned int *, unsigned int *, int, int, int, cudaStream_t);
}

namespace {

void check_cuda(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

void check_cu(CUresult error, const char *what) {
    if (error != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *text = nullptr;
        cuGetErrorName(error, &name);
        cuGetErrorString(error, &text);
        std::fprintf(stderr, "%s: %s (%s)\n", what, text ? text : "unknown", name ? name : "unknown");
        std::exit(2);
    }
}

template<typename F>
void with_context(CUcontext context, F &&f) {
    CUcontext previous = nullptr;
    check_cu(cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
    const bool pushed = previous != context;
    if (pushed) { check_cu(cuCtxPushCurrent(context), "cuCtxPushCurrent"); }
    f();
    if (pushed) {
        CUcontext popped = nullptr;
        check_cu(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
        if (popped != context) {
            std::fprintf(stderr, "context stack mismatch\n");
            std::exit(2);
        }
    }
}

template<typename T>
T *raw(BufferView<T> view) {
    return reinterpret_cast<T *>(view.native_handle()) + view.offset();
}

// Env helpers: every knob is documented in --help (see print_usage).
bool env_flag(const char *name, bool fallback = false) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') { return fallback; }
    return env[0] != '0';
}

int env_int(const char *name, int fallback) {
    const char *env = std::getenv(name);
    return env != nullptr ? std::max(0, std::atoi(env)) : fallback;
}

void print_usage(const char *exe) {
    std::printf(
        "usage: %s [backend] [num_items<=UINT_MAX] [repetitions>=5]\n"
        "       %s [num_items] [repetitions]        (implicit cuda backend)\n"
        "\n"
        "Benchmarks lcpp DeviceReduce / DeviceScan / DeviceRadixSort against a\n"
        "same-run CUB baseline on the same physical GPU and prints CSV rows:\n"
        "  operation,items,lcpp_median_ms,lcpp_min_ms,lcpp_max_ms,\n"
        "  cub_median_ms,cub_min_ms,cub_max_ms,lcpp_over_cub\n"
        "\n"
        "Environment:\n",
        exe, exe);
    struct Knob {
        const char *name;
        std::initializer_list<const char *> lines;
    };
    const Knob knobs[] = {
        {"LCPP_INTEROP=1",
         {"vk only: compile the kernels with the cuda backend and launch them",
          "inside the vk command buffer (VK_NV_cuda_kernel_launch)."}},
        {"LCPP_BENCH_BATCH=N",
         {"Non-cuda (dx/vk) host timer only: number of op invocations, each",
          "recorded into its own CommandList and committed separately, bracketed",
          "by ONE synchronize() pair. The reported time is the elapsed wall time",
          "divided by N, i.e. the per-op average including 1/N of the",
          "submit+sync cost. N=1 exposes the raw per-op host floor (~120-130 us",
          "of vk submit+sync); N>=100 amortizes it away and exposes GPU",
          "throughput -- use N>=100 for throughput tables and N=1 for host-floor",
          "tables. Default 10 (the epoch-0/1 protocol). The cuda backend ignores",
          "it and times with GPU events (or CUDA graphs with",
          "LCPP_BENCH_GRAPH=1)."}},
        {"LCPP_BENCH_POLL=1",
         {"End the timed window by signaling a stream event and polling",
          "Event::is_completed() instead of synchronize(), i.e. without the",
          "backend's worker-thread completion hand-off."}},
        {"LCPP_BENCH_INQUEUE_LIMIT=0",
         {"vk only: turn the backend's depth-2 in-flight throttle",
          "(DeviceConfig::inqueue_buffer_limit: every dispatch waits for fence-2",
          "before submitting) OFF. WARNING: measured to be a large bimodal",
          "regression on WDDM (batch=100 at 4M items: reduce 0.023 -> 0.61 ms,",
          "scan 0.085 -> 1.14 ms, radix 0.687 -> 0.97 ms; up to 7 ms/op samples",
          "at batch=8..32); it changes nothing at batch=1. Default: throttle on",
          "(the stock backend setting)."}},
        {"LCPP_BENCH_CL_OPS=K",
         {"K>=2: additionally measure K op invocations recorded into ONE caller",
          "CommandList over K distinct non-aliasing buffer sets, at one commit",
          "per sample. The vk command-reorder pass merges such iterations into",
          "barrier-free layers, so this is the pattern that hides the per-op",
          "submit floor; rows are named \"<op>[xK/CL]\"."}},
        {"LCPP_BENCH_GRAPH=1",
         {"cuda only: capture the batch into a CUDA graph and replay it."}},
    };
    for (auto &&knob : knobs) {
        auto first = true;
        for (auto *line : knob.lines) {
            std::printf("  %-26s %s\n", first ? knob.name : "", line);
            first = false;
        }
    }
}

struct Stats {
    std::vector<float> values;
    [[nodiscard]] float median() const {
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2u];
    }
    [[nodiscard]] float minimum() const { return *std::min_element(values.begin(), values.end()); }
    [[nodiscard]] float maximum() const { return *std::max_element(values.begin(), values.end()); }
};

void print_stats(std::string_view name, const Stats &lcpp, const Stats &cub, size_t num_items,
                 std::string_view suffix = {}) {
    const auto lcpp_median = lcpp.median();
    const auto cub_median  = cub.median();
    auto label = suffix.empty() ? std::string{name} : std::string{name} + std::string{suffix};
    std::printf("%s,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4fx\n",
                label.c_str(),
                num_items,
                lcpp_median,
                lcpp.minimum(),
                lcpp.maximum(),
                cub_median,
                cub.minimum(),
                cub.maximum(),
                lcpp_median / cub_median);
    std::fflush(stdout);  // keep results visible even if teardown aborts
}

// ---------------------------------------------------------------------------
// Timers for the lcpp side
// ---------------------------------------------------------------------------

struct LcppTimer {
    virtual ~LcppTimer() = default;
    virtual float measure(const char *name, const std::function<void()> &enqueue) = 0;
};

// CUDA backend: events recorded on the LC stream itself; optional graph mode.
struct CudaStreamTimer final : LcppTimer {
    CUstream stream{};
    CUcontext context{};
    CUevent start{};
    CUevent stop{};
    bool graph_mode = std::getenv("LCPP_BENCH_GRAPH") != nullptr;
    int graph_batch = std::getenv("LCPP_BENCH_BATCH") ? std::max(1, std::atoi(std::getenv("LCPP_BENCH_BATCH"))) : 1;
    std::unordered_map<std::string, CUgraphExec> graphs;

    CudaStreamTimer(CUstream s, CUcontext c) : stream{s}, context{c} {
        with_context(context, [&] {
            check_cu(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
            check_cu(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
        });
    }
    ~CudaStreamTimer() override {
        with_context(context, [&] {
            for (auto [name, graph] : graphs) { cuGraphExecDestroy(graph); }
            cuEventDestroy(start);
            cuEventDestroy(stop);
        });
    }
    float measure(const char *name, const std::function<void()> &enqueue) override {
        CUgraphExec graph_exec{};
        if (graph_mode) {
            auto it = graphs.find(name);
            if (it == graphs.end()) {
                with_context(context, [&] {
                    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize(capture)");
                    check_cu(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL), "cuStreamBeginCapture");
                    for (int batch = 0; batch < graph_batch; ++batch) { enqueue(); }
                    CUgraph graph{};
                    check_cu(cuStreamEndCapture(stream, &graph), "cuStreamEndCapture");
                    check_cu(cuGraphInstantiate(&graph_exec, graph, 0), "cuGraphInstantiate");
                    check_cu(cuGraphDestroy(graph), "cuGraphDestroy");
                    check_cu(cuGraphLaunch(graph_exec, stream), "cuGraphLaunch(warmup)");
                    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize(graph warmup)");
                });
                it = graphs.emplace(name, graph_exec).first;
            }
            graph_exec = it->second;
        }
        with_context(context, [&] { check_cu(cuEventRecord(start, stream), "cuEventRecord(start)"); });
        if (graph_mode) {
            with_context(context, [&] { check_cu(cuGraphLaunch(graph_exec, stream), "cuGraphLaunch"); });
        } else {
            enqueue();
        }
        float elapsed_ms = 0.0f;
        with_context(context, [&] {
            check_cu(cuEventRecord(stop, stream), "cuEventRecord(stop)");
            check_cu(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
            check_cu(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime");
        });
        return elapsed_ms / (graph_mode ? graph_batch : 1);
    }
};

// Non-CUDA backends (dx/vk/...): wall-clock timing around enqueue + synchronize.
// A full stream synchronize() on these backends costs ~100+ us, which would
// drown the GPU time of small ops, so we amortize: `batch` enqueues are
// submitted back-to-back and bracketed by a single synchronize pair, and the
// elapsed wall time is divided by the batch (per-op average throughput,
// comparable to how the CUB side is timed with GPU events on its own stream).
// Epoch-0/1 numbers used batch=1 (LCPP_BENCH_BATCH=1 reproduces them).
struct HostSyncTimer final : LcppTimer {
    Stream *stream;
    // Epoch C (plan §4.2): LCPP_BENCH_POLL=1 replaces the closing
    // synchronize() with a stream event + host-side completion poll.  On vk a
    // plain synchronize() spins on the stream's internal event, which is only
    // published after the backend worker thread has observed GPU completion
    // (vkWaitSemaphores) and run its notification item; polling a user event
    // keeps the same device-side fence but removes the app-thread hand-off
    // from the measured window.  Both paths wait for the same GPU work.
    Event event;
    bool poll_mode = env_flag("LCPP_BENCH_POLL");
    int batch = std::max(1, env_int("LCPP_BENCH_BATCH", 10));
    HostSyncTimer(Stream &s, Device &d) : stream(&s) {
        if (poll_mode) { event = d.create_event(); }
    }
    void wait_idle() {
        if (poll_mode) {
            *stream << event.signal();
            while (!event.is_completed()) { std::this_thread::yield(); }
        } else {
            stream->synchronize();
        }
    }
    float measure(const char *, const std::function<void()> &enqueue) override {
        wait_idle();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < batch; ++i) { enqueue(); }
        wait_idle();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count() / static_cast<float>(batch);
    }
};

// ---------------------------------------------------------------------------
// Standalone CUB baseline (runtime API only) used when the lcpp backend is
// not CUDA: CUB still runs on the same physical GPU through its own stream.
// ---------------------------------------------------------------------------

struct CubStandalone {
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    bool ok = false;

    CubStandalone() {
        ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreate(&start) == cudaSuccess &&
             cudaEventCreate(&stop) == cudaSuccess;
        if (!ok) { std::fprintf(stderr, "failed to create standalone CUDA stream/events for CUB baseline\n"); }
    }
    ~CubStandalone() {
        if (start) { cudaEventDestroy(start); }
        if (stop) { cudaEventDestroy(stop); }
        if (stream) { cudaStreamDestroy(stream); }
    }

    float measure(const std::function<void(cudaStream_t)> &enqueue) {
        cudaEventRecord(start, stream);
        enqueue(stream);
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms;
    }
};

struct CubBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    CubBuffer() = default;
    ~CubBuffer() { reset(); }
    void reset() {
        if (ptr) { cudaFree(ptr); }
        ptr = nullptr;
        bytes = 0;
    }
    void allocate(size_t b) {
        if (b <= bytes) { return; }
        reset();
        if (cudaMalloc(&ptr, b) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(%zu) failed\n", b);
            ptr = nullptr;
            bytes = 0;
            std::exit(2);
        }
        bytes = b;
    }
};

template<typename T>
void cub_upload(CubBuffer &buf, const std::vector<T> &host, cudaStream_t stream) {
    buf.allocate(host.size() * sizeof(T));
    check_cuda(cudaMemcpyAsync(buf.ptr, host.data(), host.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(H2D)");
}

template<typename T>
void cub_download(const CubBuffer &buf, std::vector<T> &host, cudaStream_t stream) {
    host.resize(buf.bytes / sizeof(T));
    check_cuda(cudaMemcpyAsync(host.data(), buf.ptr, buf.bytes, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(download)");
}

bool is_backend_name(std::string_view s) {
    return s == "cuda" || s == "dx" || s == "vk" || s == "metal" || s == "fallback" || s == "cpu";
}

}  // namespace

int main(int argc, char **argv) {
    // usage: lcpp_cub_bench [backend] [num_items<=UINT_MAX] [repetitions>=5]
    // The legacy numeric-only form "lcpp_cub_bench [num_items] [repetitions]"
    // (implicit cuda backend) is still accepted.  Env knobs: see print_usage.
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    const char *backend_name = "cuda";
    int argi = 1;
    if (argc > argi && is_backend_name(argv[argi])) {
        backend_name = argv[argi];
        ++argi;
    }
    size_t num_items = 1u << 22u;
    int repetitions  = 10;
    if (argc > argi) { num_items = static_cast<size_t>(std::strtoull(argv[argi], nullptr, 10)); }
    if (argc > argi + 1) { repetitions = std::atoi(argv[argi + 1]); }
    if (num_items == 0u || num_items > static_cast<size_t>(std::numeric_limits<uint>::max()) || repetitions < 5) {
        std::fprintf(stderr, "usage: %s [backend] [num_items<=UINT_MAX] [repetitions>=5]  (see --help)\n", argv[0]);
        return 1;
    }

    const bool is_vk = std::string_view{backend_name} == "vk";
    Context context{argv[0]};
    // GPU name for the report. On the vk backend, backend_device_names() must
    // NOT be called before device creation (volk global dispatch tables) nor
    // while a device is live, so vk simply reports "unknown".
    std::string gpu_name = "unknown";
    if (!is_vk) {
        const auto gpu_names = context.backend_device_names(backend_name);
        if (!gpu_names.empty()) { gpu_name = gpu_names.front(); }
    }
    // Epoch C (plan §4.1): expose the raw vk submit floor.  The vk backend's
    // depth-2 in-flight throttle (DeviceConfig::inqueue_buffer_limit,
    // D:/LuisaCompute/src/backends/vk/stream.cpp:1962-1966) makes every
    // dispatch wait for fence-2 before submitting, so the plan asked for it to
    // be turned off by default in the bench.  MEASURED (Epoch C, §C1 of
    // .kimix_cache/epoch_C_report.md): turning it off is a large, bimodal
    // REGRESSION on this platform (WDDM): at 4M items / batch=100 the per-op
    // wall time goes 0.023 -> 0.61 ms (reduce), 0.085 -> 1.14 ms (scan),
    // 0.687 -> 0.97 ms (radix), with individual samples up to 7 ms/op at
    // batch=8..32, while at batch=1 (a single op in flight) it changes nothing
    // (the throttle only engages from fence 3 on).  The bench therefore keeps
    // the stock setting by default and exposes the plan's knob as an opt-in
    // A/B: LCPP_BENCH_INQUEUE_LIMIT=0 turns the throttle off.
    const bool inqueue_limit = !is_vk || env_flag("LCPP_BENCH_INQUEUE_LIMIT", true);
    DeviceConfig device_config{};
    device_config.inqueue_buffer_limit = inqueue_limit;
    Device device = is_vk ? context.create_device(backend_name, &device_config)
                          : context.create_device(backend_name);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s'\n", backend_name);
        return 2;
    }
    Stream stream = device.create_stream();
    const bool is_cuda = std::string_view{backend_name} == "cuda";
    const bool poll_mode = env_flag("LCPP_BENCH_POLL");
    const int cl_ops = env_int("LCPP_BENCH_CL_OPS", 0);

    // Epoch A: opt-in vk-cuda interop fast path (env LCPP_INTEROP=1, vk only).
    // The context must outlive every lcpp module that dispatches through it and
    // be destroyed before the vk device it borrows (see interop_dispatch.h).
    auto interop_requested = [] {
        const char *env = std::getenv("LCPP_INTEROP");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    luisa::unique_ptr<InteropContext> interop;
    if (interop_requested) {
        if (std::string_view{backend_name} == "vk") {
            interop = InteropContext::create(context, device);
        } else {
            std::fprintf(stderr, "LCPP_INTEROP=1 requires the vk backend; ignored.\n");
        }
    }
    const bool interop_active = interop != nullptr;

    CUstream raw_stream = nullptr;
    CUcontext context_handle = nullptr;
    if (is_cuda) {
        raw_stream = reinterpret_cast<CUstream>(stream.native_handle());
        context_handle = reinterpret_cast<CUcontext>(device.native_handle());
        CUcontext previous_context = nullptr;
        check_cu(cuCtxGetCurrent(&previous_context), "cuCtxGetCurrent");
        (void)previous_context;
    }

    std::printf("gpu=%s backend=%s cub=%d items=%zu reps=%d interop=%d batch=%d poll=%d inqueue_limit=%d cl_ops=%d\n",
                gpu_name.c_str(),
                device.backend_name().data(), cub_bench_version(), num_items, repetitions,
                interop_active ? 1 : 0,
                is_cuda ? 0 : std::max(1, env_int("LCPP_BENCH_BATCH", 10)),
                poll_mode ? 1 : 0,
                inqueue_limit ? 1 : 0,
                cl_ops);
    std::printf("operation,items,lcpp_median_ms,lcpp_min_ms,lcpp_max_ms,cub_median_ms,cub_min_ms,cub_max_ms,lcpp_over_cub\n");

    // lcpp-side timer
    std::unique_ptr<LcppTimer> lcpp_timer;
    if (is_cuda) {
        lcpp_timer = std::make_unique<CudaStreamTimer>(raw_stream, context_handle);
    } else {
        lcpp_timer = std::make_unique<HostSyncTimer>(stream, device);
    }
    // CUB baseline: standalone CUDA stream when lcpp runs on another backend.
    CubStandalone cub;
    if (!is_cuda && !cub.ok) { return 2; }

    // =========================== DeviceReduce::Sum ===========================
    {
        auto input       = device.create_buffer<int>(num_items);
        auto lcpp_output = device.create_buffer<int>(1u);
        auto cub_output  = device.create_buffer<int>(1u);
        using ReduceBench = DeviceReduce<LCPP_REDUCE_BLOCK, 32u, LCPP_REDUCE_ITEMS>;
        auto lcpp_temp   = device.create_buffer<uint>(bytes_to_uint_count(ReduceBench::GetTempStorageBytes<int>(num_items)));
        std::vector<int> host_input(num_items);
        int expected_reduce = 0;
        for (size_t i = 0; i < num_items; ++i) {
            host_input[i] = static_cast<int>(i % 13u) - 6;
            expected_reduce += host_input[i];
        }
        stream << input.copy_from(span{host_input}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_query(raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB reduce query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(reduce temp)"); });
        } else {
            cub_upload(cub_in, host_input, cub.stream);
            cub_out.allocate(sizeof(int));
            check_cuda(cub_reduce_query(static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream, &cub_temp_bytes), "CUB reduce query");
            cub_temp.allocate(cub_temp_bytes);
        }

        ReduceBench lcpp_reduce;
        lcpp_reduce.create(device, nullptr, interop.get());
        if (interop_requested) {
            std::fprintf(stderr, "DeviceReduce interop_active=%d\n", lcpp_reduce.interop_active() ? 1 : 0);
        }
        CommandList warmup;
        lcpp_reduce.Sum(warmup, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB reduce warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_reduce_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream), "CUB reduce warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(reduce warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB reduce");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_reduce_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), s), "CUB reduce");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("reduce_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("reduce_lcpp", [&] {
                CommandList commands;
                lcpp_reduce.Sum(commands, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        int lcpp_result = 0;
        int cub_result  = 0;
        stream << lcpp_output.copy_to(span{&lcpp_result, 1u}) << synchronize();
        if (is_cuda) {
            stream << cub_output.copy_to(span{&cub_result, 1u}) << synchronize();
        } else {
            std::vector<int> tmp;
            cub_download(cub_out, tmp, cub.stream);
            cub_result = tmp[0];
        }
        if (lcpp_result != expected_reduce || cub_result != expected_reduce) {
            std::fprintf(stderr, "reduce verification failed: lcpp=%d cub=%d expected=%d\n", lcpp_result, cub_result, expected_reduce);
            return 3;
        }
        print_stats("DeviceReduce.Sum", lcpp_stats, cub_stats, num_items);
        // Epoch C (plan §4.3): K invocations of the op recorded into ONE caller
        // CommandList over K distinct (non-aliasing) buffer sets.  The vk
        // command-reorder pass merges such iterations into barrier-free layers,
        // so this measures the per-op floor of the pattern that hides the
        // per-op submit cost (batch=1, one commit per sample).
        if (cl_ops >= 2 && !is_cuda) {
            std::vector<Buffer<int>> in_k, out_k;
            std::vector<Buffer<uint>> temp_k;
            for (int k = 0; k < cl_ops; ++k) {
                in_k.emplace_back(device.create_buffer<int>(num_items));
                out_k.emplace_back(device.create_buffer<int>(1u));
                temp_k.emplace_back(device.create_buffer<uint>(bytes_to_uint_count(ReduceBench::GetTempStorageBytes<int>(num_items))));
                stream << in_k.back().copy_from(span{host_input});
            }
            CommandList cl_warm;
            for (int k = 0; k < cl_ops; ++k) {
                lcpp_reduce.Sum(cl_warm, temp_k[k].view(), in_k[k].view(), out_k[k].view(), num_items);
            }
            stream << cl_warm.commit() << synchronize();
            HostSyncTimer cl_timer{stream, device};
            cl_timer.batch = 1;
            Stats cl_stats;
            cl_stats.values.reserve(repetitions);
            for (int i = 0; i < repetitions; ++i) {
                auto total = cl_timer.measure("reduce_cl_ops", [&] {
                    CommandList commands;
                    for (int k = 0; k < cl_ops; ++k) {
                        lcpp_reduce.Sum(commands, temp_k[k].view(), in_k[k].view(), out_k[k].view(), num_items);
                    }
                    stream << commands.commit();
                });
                cl_stats.values.emplace_back(total / static_cast<float>(cl_ops));
            }
            // Exactness of every merged-layer iteration (this is also the R2
            // hazard check for K non-aliasing launches in one layer).
            auto bad = 0;
            for (int k = 0; k < cl_ops; ++k) {
                int value = 0;
                stream << out_k[k].copy_to(span{&value, 1u}) << synchronize();
                if (value != expected_reduce) { ++bad; }
            }
            print_stats("DeviceReduce.Sum", cl_stats, cub_stats, num_items,
                        "[x" + std::to_string(cl_ops) + "/CL]");
            if (bad != 0) {
                std::fprintf(stderr, "cl_ops reduce verification failed in %d/%d sets\n", bad, cl_ops);
                return 3;
            }
        }
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(reduce temp)"); });
        }
    }

    // ========================= DeviceScan::InclusiveSum =========================
    {
        auto input       = device.create_buffer<int>(num_items);
        auto lcpp_output = device.create_buffer<int>(num_items);
        auto cub_output  = device.create_buffer<int>(num_items);
        using ScanBench = DeviceScan<LCPP_SCAN_BLOCK, 32u, LCPP_SCAN_ITEMS>;
        auto lcpp_temp   = device.create_buffer<uint>(bytes_to_uint_count(ScanBench::GetTempStorageBytes<int>(num_items)));
        std::vector<int> host_input(num_items);
        std::vector<int> expected(num_items);
        int running_sum = 0;
        for (size_t i = 0u; i < num_items; ++i) {
            host_input[i] = static_cast<int>(i % 7u) - 3;
            running_sum += host_input[i];
            expected[i] = running_sum;
        }
        stream << input.copy_from(span{host_input}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_query(raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB scan query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(scan temp)"); });
        } else {
            cub_upload(cub_in, host_input, cub.stream);
            cub_out.allocate(num_items * sizeof(int));
            check_cuda(cub_scan_query(static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream, &cub_temp_bytes), "CUB scan query");
            cub_temp.allocate(cub_temp_bytes);
        }

          ScanBench lcpp_scan;
          lcpp_scan.create(device, nullptr, interop.get());
          if (interop_requested) {
              std::fprintf(stderr, "DeviceScan interop_active=%d\n", lcpp_scan.interop_active() ? 1 : 0);
          }
        CommandList warmup;
        lcpp_scan.InclusiveSum(warmup, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB scan warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_scan_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream), "CUB scan warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(scan warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB scan");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_scan_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), s), "CUB scan");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("scan_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("scan_lcpp", [&] {
                CommandList commands;
                lcpp_scan.InclusiveSum(commands, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        std::vector<int> lcpp_result(num_items);
        std::vector<int> cub_result(num_items);
        stream << lcpp_output.copy_to(span{lcpp_result}) << synchronize();
        if (is_cuda) {
            stream << cub_output.copy_to(span{cub_result}) << synchronize();
        } else {
            cub_download(cub_out, cub_result, cub.stream);
        }
        const bool lcpp_scan_ok = std::equal(lcpp_result.begin(), lcpp_result.end(), expected.begin());
        const bool cub_scan_ok = std::equal(cub_result.begin(), cub_result.end(), expected.begin());
        if (!lcpp_scan_ok || !cub_scan_ok) {
            std::fprintf(stderr, "scan verification failed\n");
            return 3;
        }
        print_stats("DeviceScan.InclusiveSum", lcpp_stats, cub_stats, num_items);
        // Epoch C §4.3: K scan invocations in one CommandList over K distinct
        // buffer sets (see the reduce block above for the rationale).
        if (cl_ops >= 2 && !is_cuda) {
            std::vector<Buffer<int>> in_k, out_k;
            std::vector<Buffer<uint>> temp_k;
            for (int k = 0; k < cl_ops; ++k) {
                in_k.emplace_back(device.create_buffer<int>(num_items));
                out_k.emplace_back(device.create_buffer<int>(num_items));
                temp_k.emplace_back(device.create_buffer<uint>(bytes_to_uint_count(ScanBench::GetTempStorageBytes<int>(num_items))));
                stream << in_k.back().copy_from(span{host_input});
            }
            CommandList cl_warm;
            for (int k = 0; k < cl_ops; ++k) {
                lcpp_scan.InclusiveSum(cl_warm, temp_k[k].view(), in_k[k].view(), out_k[k].view(), num_items);
            }
            stream << cl_warm.commit() << synchronize();
            HostSyncTimer cl_timer{stream, device};
            cl_timer.batch = 1;
            Stats cl_stats;
            cl_stats.values.reserve(repetitions);
            for (int i = 0; i < repetitions; ++i) {
                auto total = cl_timer.measure("scan_cl_ops", [&] {
                    CommandList commands;
                    for (int k = 0; k < cl_ops; ++k) {
                        lcpp_scan.InclusiveSum(commands, temp_k[k].view(), in_k[k].view(), out_k[k].view(), num_items);
                    }
                    stream << commands.commit();
                });
                cl_stats.values.emplace_back(total / static_cast<float>(cl_ops));
            }
            auto bad = 0;
            for (int k = 0; k < cl_ops; ++k) {
                std::vector<int> result(num_items);
                stream << out_k[k].copy_to(span{result}) << synchronize();
                if (!std::equal(result.begin(), result.end(), expected.begin())) { ++bad; }
            }
            print_stats("DeviceScan.InclusiveSum", cl_stats, cub_stats, num_items,
                        "[x" + std::to_string(cl_ops) + "/CL]");
            if (bad != 0) {
                std::fprintf(stderr, "cl_ops scan verification failed in %d/%d sets\n", bad, cl_ops);
                return 3;
            }
        }
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(scan temp)"); });
        }
    }

    // ========================= DeviceRadixSort::SortKeys =========================
    {
        auto keys_in       = device.create_buffer<uint>(num_items);
        auto keys_out      = device.create_buffer<uint>(num_items);
        auto lcpp_keys_out = device.create_buffer<uint>(num_items);
        using RadixBench = DeviceRadixSort<LCPP_RADIX_BLOCK, 32u, LCPP_RADIX_ITEMS>;
        auto lcpp_temp     = device.create_buffer<uint>(bytes_to_uint_count(RadixBench::GetSortKeysTempStorageBytes<uint>(static_cast<uint>(num_items))));
        std::vector<uint> host_keys(num_items);
        for (size_t i = 0; i < num_items; ++i) { host_keys[i] = static_cast<uint>((i * 2654435761ull) ^ (i >> 7u)); }
        stream << keys_in.copy_from(span{host_keys}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_query(raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB radix query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(radix temp)"); });
        } else {
            cub_upload(cub_in, host_keys, cub.stream);
            cub_out.allocate(num_items * sizeof(uint));
            check_cuda(cub_radix_query(static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, cub.stream, &cub_temp_bytes), "CUB radix query");
            cub_temp.allocate(cub_temp_bytes);
        }

          RadixBench lcpp_sort;
          lcpp_sort.create(device, nullptr, interop.get());
          if (interop_requested) {
              std::fprintf(stderr, "DeviceRadixSort interop_active=%d\n", lcpp_sort.interop_active() ? 1 : 0);
          }
        CommandList warmup;
        lcpp_sort.SortKeys(warmup, lcpp_temp.view(), keys_in.view(), lcpp_keys_out.view(), static_cast<uint>(num_items));
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream)), "CUB radix warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_radix_run(cub_temp.ptr, cub_temp_bytes, static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, cub.stream), "CUB radix warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(radix warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream)), "CUB radix");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_radix_run(cub_temp.ptr, cub_temp_bytes, static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, s), "CUB radix");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("radix_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("radix_lcpp", [&] {
                CommandList commands;
                lcpp_sort.SortKeys(commands, lcpp_temp.view(), keys_in.view(), lcpp_keys_out.view(), static_cast<uint>(num_items));
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        std::vector<uint> lcpp_result(num_items);
        std::vector<uint> cub_result(num_items);
        stream << lcpp_keys_out.copy_to(span{lcpp_result}) << synchronize();
        if (is_cuda) {
            stream << keys_out.copy_to(span{cub_result}) << synchronize();
        } else {
            cub_download(cub_out, cub_result, cub.stream);
        }
        auto expected = host_keys;
        std::sort(expected.begin(), expected.end());
        if (!std::is_sorted(lcpp_result.begin(), lcpp_result.end()) || !std::is_sorted(cub_result.begin(), cub_result.end()) ||
            lcpp_result != cub_result || lcpp_result != expected) {
            std::fprintf(stderr, "radix verification failed\n");
            return 3;
        }
        print_stats("DeviceRadixSort.SortKeys", lcpp_stats, cub_stats, num_items);
        // Epoch C §4.3: K radix sorts in one CommandList over K distinct buffer
        // sets -- the 12-launch op where the reorder pass has the most to merge
        // (4 init launches become one barrier-free layer, the one-sweep launches
        // of different iterations no longer serialize)...
        if (cl_ops >= 2 && !is_cuda) {
            std::vector<Buffer<uint>> keys_in_k, keys_out_k, temp_k;
            for (int k = 0; k < cl_ops; ++k) {
                keys_in_k.emplace_back(device.create_buffer<uint>(num_items));
                keys_out_k.emplace_back(device.create_buffer<uint>(num_items));
                temp_k.emplace_back(device.create_buffer<uint>(bytes_to_uint_count(RadixBench::GetSortKeysTempStorageBytes<uint>(static_cast<uint>(num_items)))));
                stream << keys_in_k.back().copy_from(span{host_keys});
            }
            CommandList cl_warm;
            for (int k = 0; k < cl_ops; ++k) {
                lcpp_sort.SortKeys(cl_warm, temp_k[k].view(), keys_in_k[k].view(), keys_out_k[k].view(), static_cast<uint>(num_items));
            }
            stream << cl_warm.commit() << synchronize();
            HostSyncTimer cl_timer{stream, device};
            cl_timer.batch = 1;
            Stats cl_stats;
            cl_stats.values.reserve(repetitions);
            for (int i = 0; i < repetitions; ++i) {
                auto total = cl_timer.measure("radix_cl_ops", [&] {
                    CommandList commands;
                    for (int k = 0; k < cl_ops; ++k) {
                        lcpp_sort.SortKeys(commands, temp_k[k].view(), keys_in_k[k].view(), keys_out_k[k].view(), static_cast<uint>(num_items));
                    }
                    stream << commands.commit();
                });
                cl_stats.values.emplace_back(total / static_cast<float>(cl_ops));
            }
            auto bad = 0;
            for (int k = 0; k < cl_ops; ++k) {
                std::vector<uint> result(num_items);
                stream << keys_out_k[k].copy_to(span{result}) << synchronize();
                if (result != expected) { ++bad; }
            }
            print_stats("DeviceRadixSort.SortKeys", cl_stats, cub_stats, num_items,
                        "[x" + std::to_string(cl_ops) + "/CL]");
            if (bad != 0) {
                std::fprintf(stderr, "cl_ops radix verification failed in %d/%d sets\n", bad, cl_ops);
                return 3;
            }
        }
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(radix temp)"); });
        }
    }

    return 0;
}
