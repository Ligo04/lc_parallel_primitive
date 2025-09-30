
#include "lc_parallel_primitive/block/block_reduce.h"
#include "lc_parallel_primitive/runtime/core.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include <lc_parallel_primitive/parallel_primitive.h>
#include <boost/ut.hpp>
#include <random>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;

int main(int argc, char* argv[])
{
    // log_level_info();
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
    // std::mt19937 rng(123);  // 固定种子
    // std::shuffle(input_data.begin(), input_data.end(), rng);

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
    LUISA_INFO("Block Reduce: {} {}", result[0], result[1]);

    // Kernel1D block_reduce_test =
    //     [&](BufferVar<float> source, BufferVar<float> result, Var<float> x) noexcept
    // { BlockReduce<float>().Sum(source, result, x); };
}