// tests/cub_bench_kernels.cu
// CUDA CUB baseline wrappers for tests/lcpp_cub_bench.cpp.
// Compiled by nvcc; exposes plain C entry points so the benchmark can link
// against NVIDIA CUB (bundled with the CUDA toolkit as part of CCCL) without
// including any CUDA headers in the C++ translation unit.

#include <cuda_runtime_api.h>

#include <cstddef>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

extern "C" {

// Version marker so the benchmark can report which baseline it linked against.
int cub_bench_version() { return 3; }

cudaError_t cub_reduce_query(const int *d_in,
                             int *d_out,
                             int num_items,
                             cudaStream_t stream,
                             size_t *temp_bytes) {
    return cub::DeviceReduce::Sum(nullptr, *temp_bytes, d_in, d_out, num_items, stream);
}

cudaError_t cub_reduce_run(void *d_temp,
                           size_t temp_bytes,
                           const int *d_in,
                           int *d_out,
                           int num_items,
                           cudaStream_t stream) {
    return cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, num_items, stream);
}

cudaError_t cub_scan_query(const int *d_in,
                           int *d_out,
                           int num_items,
                           cudaStream_t stream,
                           size_t *temp_bytes) {
    return cub::DeviceScan::InclusiveSum(nullptr, *temp_bytes, d_in, d_out, num_items, stream);
}

cudaError_t cub_scan_run(void *d_temp,
                         size_t temp_bytes,
                         const int *d_in,
                         int *d_out,
                         int num_items,
                         cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, num_items, stream);
}

cudaError_t cub_radix_query(const unsigned int *d_in,
                            unsigned int *d_out,
                            int num_items,
                            int begin_bit,
                            int end_bit,
                            cudaStream_t stream,
                            size_t *temp_bytes) {
    return cub::DeviceRadixSort::SortKeys(nullptr, *temp_bytes, d_in, d_out, num_items, begin_bit, end_bit,
                                          stream);
}

cudaError_t cub_radix_run(void *d_temp,
                          size_t temp_bytes,
                          const unsigned int *d_in,
                          unsigned int *d_out,
                          int num_items,
                          int begin_bit,
                          int end_bit,
                          cudaStream_t stream) {
    return cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, num_items,
                                          begin_bit, end_bit, stream);
}
}
