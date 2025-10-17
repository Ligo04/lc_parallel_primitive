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

    constexpr int32_t array_size = 256;

    auto               in_buffer  = device.create_buffer<int32>(array_size);
    auto               out_buffer = device.create_buffer<int32>(1);
    std::vector<int32> result(1);

    std::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }
    // std::mt19937 rng(114521);  // 固定种子
    // std::shuffle(input_data.begin(), input_data.end(), rng);

    CommandList cmdlist;
    Stream      stream = device.create_stream();
    stream << in_buffer.copy_from(input_data.data()) << synchronize();

    // reducer.Sum(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());

    // stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    // LUISA_INFO("Result (0+1+2+...+{}): {}", (array_size - 1), ((array_size - 1) * array_size) / 2);
    // LUISA_INFO("Reduce: {}", result[0]);

    // "reduce"_test = [&]
    // { expect(((array_size - 1) * array_size) / 2 == result[0]); };

    // //reduce(min)
    // reducer.Min(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());
    // stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    // LUISA_INFO("Result Min(0-{}): {}",
    //            (array_size - 1),
    //            *std::min_element(input_data.begin(), input_data.end()));
    // LUISA_INFO("Reduce Min: {}", result[0]);
    // "reduce min"_test = [&]
    // {
    //     expect(*std::min_element(input_data.begin(), input_data.end()) == result[0]);
    // };

    // // reduce(max)
    // reducer.Max(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());
    // stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    // LUISA_INFO("Result Max(0-1023): {}",
    //            *std::max_element(input_data.begin(), input_data.end()));
    // LUISA_INFO("Reduce Max: {}", result[0]);
    // "reduce max"_test = [&]
    // {
    //     expect(*std::max_element(input_data.begin(), input_data.end()) == result[0]);
    // };

    // auto index_out_buffer = device.create_buffer<luisa::uint>(1);
    // std::vector<luisa::uint> index_result(1);
    // reducer.ArgMin(cmdlist,
    //                stream,
    //                in_buffer.view(),
    //                out_buffer.view(),
    //                index_out_buffer.view(),
    //                in_buffer.size());

    // stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    // stream << index_out_buffer.copy_to(index_result.data()) << synchronize();  // 输出结果


    // LUISA_INFO("Index ArgMin: {}",
    //            std::min_element(input_data.begin(), input_data.end())
    //                - input_data.begin());
    // LUISA_INFO("Index ArgMin(reduce): {}", index_result[0]);

    // "reduce argmin"_test = [&]
    // {
    //     expect((std::min_element(input_data.begin(), input_data.end())
    //             - input_data.begin())
    //            == index_result[0]);
    // };

    // // reduce(argmax)
    // reducer.ArgMax(cmdlist,
    //                stream,
    //                in_buffer.view(),
    //                out_buffer.view(),
    //                index_out_buffer.view(),
    //                in_buffer.size());

    // stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    // stream << index_out_buffer.copy_to(index_result.data()) << synchronize();  // 输出结果


    // LUISA_INFO("Index ArgMax: {}",
    //            std::max_element(input_data.begin(), input_data.end())
    //                - input_data.begin());
    // LUISA_INFO("Index ArgMax(reduce): {}", index_result[0]);

    // "reduce argmax"_test = [&]
    // {
    //     expect((std::max_element(input_data.begin(), input_data.end())
    //             - input_data.begin())
    //            == index_result[0]);
    // };


    //reduce by key

    auto key_buffer   = device.create_buffer<int32>(array_size);
    auto value_buffer = device.create_buffer<int32>(array_size);

    constexpr int items_per_segment = 100;
    const int segments = (array_size + items_per_segment - 1) / items_per_segment;  // 向上取整

    std::vector<int32> input_keys(array_size);
    for(auto i = 0; i < array_size; i++)
    {
        input_keys[i] = i / items_per_segment;  // 每 100 个元素一组
    }

    LUISA_INFO("Array size: {}, Items per segment: {}, Total segments: {}",
               array_size,
               items_per_segment,
               segments);

    auto unique_keys_buffer = device.create_buffer<int32>(segments);
    auto aggregates_buffer  = device.create_buffer<int32>(segments);
    auto num_runs_buffer    = device.create_buffer<luisa::uint>(1);

    stream << key_buffer.copy_from(input_keys.data()) << synchronize();
    stream << value_buffer.copy_from(input_data.data()) << synchronize();


    reducer.ReduceByKey(
        cmdlist,
        stream,
        key_buffer.view(),
        value_buffer.view(),
        unique_keys_buffer.view(),
        aggregates_buffer.view(),
        num_runs_buffer.view(),
        [](const Var<int32>& a, const Var<int32>& b) { return a + b; },
        in_buffer.size());

    std::vector<int32>       unique_keys(segments);
    std::vector<int32>       aggregates(segments);
    std::vector<luisa::uint> num_runs(1);
    stream << unique_keys_buffer.copy_to(unique_keys.data()) << synchronize();  // 输出结果
    stream << aggregates_buffer.copy_to(aggregates.data()) << synchronize();  // 输出结果
    stream << num_runs_buffer.copy_to(num_runs.data()) << synchronize();  // 输出结果

    LUISA_INFO("Reduce By Key: num_runs: {}", num_runs[0]);
    "reduce by key"_test = [&]
    {
        expect(segments == num_runs[0]);
        for(auto i = 0; i < segments; i++)
        {
            auto expected_sum = 0;
            for(auto j = i * items_per_segment;
                j < (i + 1) * items_per_segment && j < array_size;
                j++)
            {
                expected_sum += input_data[j];
            }
            LUISA_INFO("Key: {}, Expected Aggregate: {}", i, expected_sum);
            LUISA_INFO("UniqueKey: {}, AggregateValue: {}", unique_keys[i], aggregates[i]);
            // expect(expected_sum == aggregates[i]);
        }
    };

    return 0;
}