
#include "lc_parallel_primitive/block/block_reduce.h"
#include "lc_parallel_primitive/block/block_scan.h"
#include "lc_parallel_primitive/runtime/core.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include <lc_parallel_primitive/parallel_primitive.h>
#include <boost/ut.hpp>
#include <numeric>
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

    DeviceReduce reducer;
    reducer.create(device);
    constexpr int      BLOCKSIZE  = 256;
    constexpr int      N          = 512;
    auto               in_buffer  = device.create_buffer<int32>(N);
    auto               out_buffer = device.create_buffer<int32>(N / 256);
    std::vector<int32> result(N / 256);

    std::vector<int32> input_data(N);
    for(int i = 0; i < N; i++)
    {
        input_data[i] = i;
    }

    stream << in_buffer.copy_from(input_data.data()) << synchronize();

    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> block_reduce_shader = nullptr;
    lazy_compile(device,
                 block_reduce_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(BLOCKSIZE);
                     Int thid = Int(block_size().x * block_id().x + thread_id().x);

                     Int thread_data = def(0);
                     $if(thid < n)
                     {
                         thread_data = arr_in.read(thid);
                     };
                     Int aggregate = BlockReduce<int>().Sum(thread_data);
                     $if(thread_id().x == 0)
                     {
                         arr_out.write(block_id().x, aggregate);
                     };
                 });

    stream << (*block_reduce_shader)(in_buffer.view(), out_buffer.view(), N).dispatch(N);
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    LUISA_INFO("Result (0+1+2+...+255) : {}  (255+255+257+...+511):{}",
               (255 * 256) / 2,
               (511 * 512) / 2 - (255 * 256) / 2);
    for(auto& data : result)
    {
        LUISA_INFO("Block Reduce: {}", data);
    }

    stream << in_buffer.copy_from(input_data.data()) << synchronize();
    auto               scan_out_buffer = device.create_buffer<int32>(N);
    std::vector<int32> scan_result(N);
    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> block_scan_shader = nullptr;
    lazy_compile(device,
                 block_scan_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(BLOCKSIZE);
                     Int thid = Int(block_size().x * block_id().x + thread_id().x);

                     Int thread_data = def(0);
                     $if(thid < n)
                     {
                         thread_data = arr_in.read(thid);
                     };
                     Int scanned_data;
                     BlockScan<int>().ExclusiveSum(thread_data, scanned_data);
                     $if(thid < n)
                     {
                         arr_out.write(thid, scanned_data);
                     };
                 });


    stream << (*block_scan_shader)(in_buffer.view(), scan_out_buffer.view(), N).dispatch(N);
    stream << scan_out_buffer.copy_to(scan_result.data()) << synchronize();  // 输出结果

    std::vector<int> exclusive_scan_result(input_data.size());
    std::exclusive_scan(
        input_data.begin(), input_data.end(), exclusive_scan_result.begin(), 0);


    LUISA_INFO("exclusive_scan Result:");
    for(auto& data : exclusive_scan_result)
    {
        std::cout << data << " ";
    }
    std::cout << std::endl;

    LUISA_INFO("Scan Result:");
    for(auto& data : scan_result)
    {
        std::cout << data << " ";
    }
    std::cout << std::endl;
}