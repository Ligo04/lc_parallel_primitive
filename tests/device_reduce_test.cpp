// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */

#include "luisa/dsl/var.h"
#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <random>
#include <vector>
#include <boost/ut.hpp>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;

int main(int argc, char* argv[])
{
    log_level_verbose();

    luisa::string_view ctx_dir;
    luisa::string_view backend_name;
#ifdef _WIN32
    backend_name = "dx";
#elif __APPLE__
    backend_name = "metal";
#else
    backend_name = "cuda";
#endif

    // Parse command-line arguments: --ctx=./dir --backend=dx
    for (int i = 1; i < argc; ++i) {
        luisa::string_view arg(argv[i]);
        if (arg.starts_with("--ctx=")) {
            ctx_dir = arg.substr(6);
            LUISA_INFO("Use context directory from command-line: {}", ctx_dir);
        } else if (arg.starts_with("--backend=")) {
            backend_name = arg.substr(10);
            LUISA_INFO("Use backend from command-line: {}", backend_name);
        }
    }

    Context context{ctx_dir.empty() ? argv[0] : ctx_dir};
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream();

    constexpr int32_t array_size = 1 << 12;
    DeviceReduce<>    reducer;
    reducer.create(device, &stream);
    luisa::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }
    std::mt19937 rng(114521);  // 固定种子
    std::shuffle(input_data.begin(), input_data.end(), rng);


    using Type4Byte = uint;
    "reduce"_test   = [&]
    {
        for(uint loop = 0; loop < 24; ++loop)
        {
            uint num_items = 1 << loop;
            num_items += 1;
            Buffer<Type4Byte>      d_input  = device.create_buffer<Type4Byte>(num_items);
            Buffer<Type4Byte>      d_output = device.create_buffer<Type4Byte>(1);
            std::vector<Type4Byte> host_input(num_items, 1);
            stream << d_input.copy_from(host_input.data()) << synchronize();

            // CUB-style: first get temp storage size, then create buffer
            size_t temp_bytes = DeviceReduce<>::GetTempStorageBytes<Type4Byte>(num_items);
            auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

            CommandList cmdlist;
            reducer.Sum(cmdlist, temp_buffer.view(), d_input.view(), d_output.view(), num_items);
            stream << cmdlist.commit() << synchronize();
            std::vector<Type4Byte> host_output(1);
            stream << d_output.copy_to(host_output.data()) << synchronize();

            LUISA_INFO("Reduce: {}", host_output[0]);
            expect(host_output[0] == num_items);
        }
    };

    "reduce_transform"_test = [&]
    {
        luisa::vector<int32> result(1);
        auto                 in_buffer  = device.create_buffer<int32>(array_size);
        auto                 out_buffer = device.create_buffer<int32>(1);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        auto max_reduce_op = [](const Var<int>& a, const Var<int>& b) noexcept
        { return compute::max(a, b); };

        auto square_op = [](const Var<int>& x) noexcept { return x * x; };

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceReduce<>::GetTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.TransformReduce(
            cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), in_buffer.size(), max_reduce_op, square_op, 0);
        stream << cmdlist.commit() << synchronize();
        stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果

        LUISA_INFO("Reduce(square) (0,1,4,...+{}): {}", (array_size - 1), (array_size - 1) * (array_size - 1));
        LUISA_INFO("Result(square): {}", result[0]);
        expect((array_size - 1) * (array_size - 1) == result[0]);
    };

    //reduce(min)
    "reduce min"_test = [&]
    {
        auto                 in_buffer  = device.create_buffer<int32>(array_size);
        auto                 out_buffer = device.create_buffer<int32>(1);
        luisa::vector<int32> result(1);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceReduce<>::GetTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.Min(cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), in_buffer.size());
        stream << cmdlist.commit() << synchronize();
        stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
        LUISA_INFO("Result Min(0-{}): {}",
                   (array_size - 1),
                   *std::min_element(input_data.begin(), input_data.end()));
        LUISA_INFO("Reduce Min: {}", result[0]);
        expect(*std::min_element(input_data.begin(), input_data.end()) == result[0]);
    };

    // reduce(max)
    "reduce_max"_test = [&]
    {
        auto                 in_buffer  = device.create_buffer<int32>(array_size);
        auto                 out_buffer = device.create_buffer<int32>(1);
        luisa::vector<int32> result(1);

        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceReduce<>::GetTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.Max(cmdlist,
                    temp_buffer.view(),
                    in_buffer.view(),
                    out_buffer.view(),
                    in_buffer.size());
        stream << cmdlist.commit() << synchronize();
        stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
        LUISA_INFO("Result Max(0-1023): {}", *std::max_element(input_data.begin(), input_data.end()));
        LUISA_INFO("Reduce Max: {}", result[0]);
        expect(*std::max_element(input_data.begin(), input_data.end()) == result[0]);
    };


    "reduce argmin"_test = [&]
    {
        auto in_buffer  = device.create_buffer<int32>(array_size);
        auto out_buffer = device.create_buffer<int32>(1);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        auto                       index_out_buffer = device.create_buffer<luisa::uint>(1);
        luisa::vector<int32>       result(1);
        luisa::vector<luisa::uint> index_result(1);

        // CUB-style: get temp storage size for ArgMin
        size_t temp_bytes = DeviceReduce<>::GetArgTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.ArgMin(
            cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), index_out_buffer.view(), in_buffer.size());
        stream << cmdlist.commit() << synchronize();
        stream << out_buffer.copy_to(result.data()) << synchronize();              // 输出结果
        stream << index_out_buffer.copy_to(index_result.data()) << synchronize();  // 输出结果
        LUISA_INFO("result index:{}, value: {}", index_result[0], result[0]);

        LUISA_INFO("Index ArgMin: {}",
                   std::min_element(input_data.begin(), input_data.end()) - input_data.begin());
        LUISA_INFO("Index ArgMin(reduce): {}", index_result[0]);
        expect((std::min_element(input_data.begin(), input_data.end()) - input_data.begin()) == index_result[0]);
    };

    // reduce(argmax)
    "reduce argmax"_test = [&]
    {
        auto in_buffer  = device.create_buffer<int32>(array_size);
        auto out_buffer = device.create_buffer<int32>(1);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        auto                       index_out_buffer = device.create_buffer<luisa::uint>(1);
        luisa::vector<int32>       result(1);
        luisa::vector<luisa::uint> index_result(1);

        // CUB-style: get temp storage size for ArgMax
        size_t temp_bytes = DeviceReduce<>::GetArgTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.ArgMax(
            cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), index_out_buffer.view(), in_buffer.size());
        stream << cmdlist.commit() << synchronize();
        stream << out_buffer.copy_to(result.data()) << synchronize();
        stream << index_out_buffer.copy_to(index_result.data()) << synchronize();


        LUISA_INFO("Index ArgMax: {}",
                   std::max_element(input_data.begin(), input_data.end()) - input_data.begin());
        LUISA_INFO("Index ArgMax(reduce): {}", index_result[0]);

        expect((std::max_element(input_data.begin(), input_data.end()) - input_data.begin()) == index_result[0]);
    };


    // reduce by key
    "reduce_by_key"_test = [&]
    {
        auto key_buffer   = device.create_buffer<int32>(array_size);
        auto value_buffer = device.create_buffer<int32>(array_size);

        constexpr int items_per_segment = 100;
        const int segments = (array_size + items_per_segment - 1) / items_per_segment;  // 向上取整

        luisa::vector<int32> input_keys(array_size);
        for(auto i = 0; i < array_size; i++)
        {
            input_keys[i] = i / items_per_segment;  // 每 100 个元素一组
        }

        LUISA_INFO("Array size: {}, Items per segment: {}, Total segments: {}", array_size, items_per_segment, segments);

        auto unique_keys_buffer = device.create_buffer<int32>(segments);
        auto aggregates_buffer  = device.create_buffer<int32>(segments);
        auto num_runs_buffer    = device.create_buffer<luisa::uint>(1);

        stream << key_buffer.copy_from(input_keys.data()) << synchronize();
        stream << value_buffer.copy_from(input_data.data()) << synchronize();

        // CUB-style: get temp storage size for ReduceByKey
        size_t temp_bytes = DeviceReduce<>::GetReduceByKeyTempStorageBytes<int32, int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList cmdlist;
        reducer.ReduceByKey(
            cmdlist,
            temp_buffer.view(),
            key_buffer.view(),
            value_buffer.view(),
            unique_keys_buffer.view(),
            aggregates_buffer.view(),
            num_runs_buffer.view(),
            [](const Var<int32>& a, const Var<int32>& b) { return a + b; },
            key_buffer.size());
        stream << cmdlist.commit() << synchronize();
        luisa::vector<int32>       unique_keys(segments);
        luisa::vector<int32>       aggregates(segments);
        luisa::vector<luisa::uint> num_runs(1);
        stream << unique_keys_buffer.copy_to(unique_keys.data()) << synchronize();  // 输出结果
        stream << aggregates_buffer.copy_to(aggregates.data()) << synchronize();    // 输出结果
        stream << num_runs_buffer.copy_to(num_runs.data()) << synchronize();        // 输出结果

        LUISA_INFO("Reduce By Key: num_runs: {}", num_runs[0]);

        expect(segments == num_runs[0]);
        for(auto i = 0; i < segments; i++)
        {
            auto expected_sum = 0;
            for(auto j = i * items_per_segment; j < (i + 1) * items_per_segment && j < array_size; j++)
            {
                expected_sum += input_data[j];
            }
            LUISA_INFO("Key: {}, Expected Aggregate: {}", i, expected_sum);
            LUISA_INFO("UniqueKey: {}, AggregateValue: {}", unique_keys[i], aggregates[i]);
            expect(expected_sum == aggregates[i]);
        }
    };
}
