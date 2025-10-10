// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */

#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lc_parallel_primitive/parallel_primitive.h>
#include <random>
#include <vector>
#include <boost/ut.hpp>
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

    DeviceReduce reducer;
    reducer.create(device);

    constexpr int32_t array_size = 10240;

    auto               in_buffer  = device.create_buffer<int32>(array_size);
    auto               out_buffer = device.create_buffer<int32>(1);
    std::vector<int32> result(1);

    std::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }
    std::mt19937 rng(123);  // 固定种子
    std::shuffle(input_data.begin(), input_data.end(), rng);

    CommandList cmdlist;
    Stream      stream = device.create_stream();
    stream << in_buffer.copy_from(input_data.data()) << synchronize();

    reducer.Sum(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());

    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    LUISA_INFO("Result (0+1+2+...+{}): {}", (array_size - 1), ((array_size - 1) * array_size) / 2);
    LUISA_INFO("Reduce: {}", result[0]);

    "reduce"_test = [&]
    { expect(((array_size - 1) * array_size) / 2 == result[0]); };

    //reduce(min)
    reducer.Min(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    LUISA_INFO("Result Min(0-{}): {}",
               (array_size - 1),
               *std::min_element(input_data.begin(), input_data.end()));
    LUISA_INFO("Reduce Min: {}", result[0]);
    "reduce min"_test = [&]
    {
        expect(*std::min_element(input_data.begin(), input_data.end()) == result[0]);
    };

    // reduce(max)
    reducer.Max(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    LUISA_INFO("Result Max(0-1023): {}",
               *std::max_element(input_data.begin(), input_data.end()));
    LUISA_INFO("Reduce Max: {}", result[0]);
    "reduce max"_test = [&]
    {
        expect(*std::max_element(input_data.begin(), input_data.end()) == result[0]);
    };

    auto index_out_buffer = device.create_buffer<luisa::uint>(1);
    std::vector<luisa::uint> index_result(1);
    reducer.ArgMin(cmdlist,
                   stream,
                   in_buffer.view(),
                   out_buffer.view(),
                   index_out_buffer.view(),
                   in_buffer.size());

    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    stream << index_out_buffer.copy_to(index_result.data()) << synchronize();  // 输出结果


    LUISA_INFO("Index ArgMin: {}",
               std::min_element(input_data.begin(), input_data.end())
                   - input_data.begin());
    LUISA_INFO("Index ArgMin(reduce): {}", index_result[0]);

    "reduce argmin"_test = [&]
    {
        expect((std::min_element(input_data.begin(), input_data.end())
                - input_data.begin())
               == index_result[0]);
    };

    // reduce(argmax)
    reducer.ArgMax(cmdlist,
                   stream,
                   in_buffer.view(),
                   out_buffer.view(),
                   index_out_buffer.view(),
                   in_buffer.size());

    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    stream << index_out_buffer.copy_to(index_result.data()) << synchronize();  // 输出结果


    LUISA_INFO("Index ArgMax: {}",
               std::max_element(input_data.begin(), input_data.end())
                   - input_data.begin());
    LUISA_INFO("Index ArgMax(reduce): {}", index_result[0]);

    "reduce argmax"_test = [&]
    {
        expect((std::max_element(input_data.begin(), input_data.end())
                - input_data.begin())
               == index_result[0]);
    };
    return 0;
}