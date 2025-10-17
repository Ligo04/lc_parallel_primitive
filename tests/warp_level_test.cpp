
#include "lc_parallel_primitive/block/block_reduce.h"
#include "lc_parallel_primitive/block/block_scan.h"
#include "lc_parallel_primitive/runtime/core.h"
#include "lc_parallel_primitive/warp/warp_reduce.h"
#include "lc_parallel_primitive/warp/warp_scan.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include <cstddef>
#include <lc_parallel_primitive/parallel_primitive.h>
#include <boost/ut.hpp>
#include <numeric>
#include <vector>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;

int main(int argc, char* argv[])
{
    log_level_verbose();

    Context context{argv[1]};
#ifdef _WIN32
    Device device = context.create_device("cuda");
#elif __APPLE__
    Device device = context.create_device("metal");
#endif
    CommandList cmdlist;
    Stream      stream = device.create_stream();


    constexpr size_t WARP_SIZE  = 32;
    constexpr size_t array_size = 256;
    constexpr size_t BLOCK_SIZE = 256;

    auto in_buffer = device.create_buffer<int32>(array_size);
    auto reduce_out_buffer = device.create_buffer<int32>(array_size / WARP_SIZE);
    std::vector<int32> result(array_size / WARP_SIZE);

    std::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }

    stream << in_buffer.copy_from(input_data.data()) << synchronize();

    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_reduce_test_shader = nullptr;
    lazy_compile(device,
                 warp_reduce_test_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(BLOCK_SIZE);
                     luisa::compute::set_warp_size(WARP_SIZE);
                     Int thid = Int(block_size().x * block_id().x + thread_id().x);
                     Int thread_data = def(0);
                     $if(thid < n)
                     {
                         thread_data = arr_in.read(thid);
                     };
                     Int result = WarpReduce<int>().Reduce(thread_data,
                                                           [](const Var<int>& a,
                                                              const Var<int>& b) noexcept
                                                           { return a + b; });
                     // device_log("thread_id: {}, thid: {}, result: {}", thread_id().x, thid, result);
                     $if(compute::warp_lane_id() == 0)
                     {
                         arr_out.write(thid / UInt(WARP_SIZE), result);
                     };
                 });

    stream << (*warp_reduce_test_shader)(in_buffer.view(), reduce_out_buffer.view(), array_size)
                  .dispatch(array_size);
    stream << reduce_out_buffer.copy_to(result.data()) << synchronize();  // 输出结果

    "test_warp_reduce"_test = [&]
    {
        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            auto index_result = std::accumulate(input_data.begin() + i * WARP_SIZE,
                                                input_data.begin() + (i + 1) * WARP_SIZE,
                                                0);
            // LUISA_INFO("index: {}, index_result: {}, warp_reduce_result: {}",
            //            i,
            //            index_result,
            //            result[i]);
            expect(result[i] == index_result);
        }
    };

    auto scan_out_buffer = device.create_buffer<int32>(array_size);
    std::vector<int32> scan_result(array_size);
    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_ex_scan_test_shader = nullptr;
    lazy_compile(device,
                 warp_ex_scan_test_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(BLOCK_SIZE);
                     luisa::compute::set_warp_size(WARP_SIZE);
                     Int thid = Int(block_size().x * block_id().x + thread_id().x);
                     Int thread_data = def(0);
                     $if(thid < n)
                     {
                         thread_data = arr_in.read(thid);
                     };
                     Int output_block_scan;
                     Int warp_aggregate;
                     output_block_scan = warp_prefix_sum(thread_data);
                     WarpScan<int>().ExclusiveScan(
                         thread_data,
                         output_block_scan,
                         warp_aggregate,
                         [](const Var<int>& a, const Var<int>& b) noexcept
                         { return a + b; },
                         Int(0));
                     $if(thid < n)
                     {
                         arr_out.write(thid, output_block_scan);
                     };
                 });

    stream << (*warp_ex_scan_test_shader)(in_buffer.view(), scan_out_buffer.view(), array_size)
                  .dispatch(array_size);
    stream << scan_out_buffer.copy_to(scan_result.data()) << synchronize();  // 输出结果

    "test_warp_ex_scan"_test = [&]
    {
        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            std::vector<int> exclusive_scan_result(WARP_SIZE);
            std::exclusive_scan(input_data.begin() + i * WARP_SIZE,
                                input_data.begin() + (i + 1) * WARP_SIZE,
                                exclusive_scan_result.begin(),
                                0);
            for(auto j = 0; j < WARP_SIZE; ++j)
            {
                // LUISA_INFO("index: {}, index_result: {}, warp_reduce_result: {}",
                //            i * WARP_SIZE + j,
                //            exclusive_scan_result[j],
                //            scan_result[i * WARP_SIZE + j]);
                expect(exclusive_scan_result[j] == scan_result[i * WARP_SIZE + j]);
            }
        };
    };
}