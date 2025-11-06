/*
 * @Author: Ligo 
 * @Date: 2025-11-06 14:30:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:23:56
 */


#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <numeric>
#include <random>
#include <boost/ut.hpp>
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
    Stream      stream = device.create_stream();
    CommandList cmdlist;

    constexpr int32_t array_size       = 10240;
    constexpr int32_t BLOCK_SIZE       = 128;
    constexpr int32_t ITEMS_PER_THREAD = 2;
    constexpr int32_t WARP_NUMS        = 32;

    DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD> scanner;
    scanner.create(device);

    luisa::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }
    // std::mt19937 rng(114521);  // 固定种子
    // std::shuffle(input_data.begin(), input_data.end(), rng);

    "exclusive_scan"_test = [&]
    {
        auto in_buffer  = device.create_buffer<int32>(array_size);
        auto out_buffer = device.create_buffer<int32>(array_size);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        scanner.ExclusiveSum(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());

        luisa::vector<int32> result(array_size);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        luisa::vector<int32> expected(array_size);
        std::exclusive_scan(input_data.begin(), input_data.end(), expected.begin(), 0);

        for(auto i = 0; i < array_size; i++)
        {
            LUISA_INFO("exclusiv {}: {} - (expected): {}", i, result[i], expected[i]);
            expect(result[i] == expected[i]);
        }
    };

    "inclusive_scan"_test = [&]
    {
        auto in_buffer  = device.create_buffer<int32>(array_size);
        auto out_buffer = device.create_buffer<int32>(array_size);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        scanner.InclusiveSum(cmdlist, stream, in_buffer.view(), out_buffer.view(), in_buffer.size());

        luisa::vector<int32> result(array_size);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        luisa::vector<int32> expected(array_size);
        std::inclusive_scan(input_data.begin(), input_data.end(), expected.begin());

        for(auto i = 0; i < array_size; i++)
        {
            LUISA_INFO("inclusive {}: {} - (expected): {}", i, result[i], expected[i]);
            expect(result[i] == expected[i]);
        }
    };

    "exclusive_scan"_test = [&]
    {
        auto key_buffer   = device.create_buffer<int32>(array_size);
        auto value_buffer = device.create_buffer<int32>(array_size);

        constexpr int items_per_segment = 100;
        const int     segments          = (array_size + items_per_segment - 1) / items_per_segment;

        luisa::vector<int32> input_keys(array_size);
        for(auto i = 0; i < array_size; i++)
        {
            input_keys[i] = i / items_per_segment;
        }

        LUISA_INFO("Array size: {}, Items per segment: {}, Total segments: {}", array_size, items_per_segment, segments);

        stream << key_buffer.copy_from(input_keys.data()) << synchronize();
        stream << value_buffer.copy_from(input_data.data()) << synchronize();

        auto value_out_buffer = device.create_buffer<int32>(array_size);
        scanner.ExclusiveSumByKey(cmdlist,
                                  stream,
                                  key_buffer.view(),
                                  value_buffer.view(),
                                  value_out_buffer.view(),
                                  key_buffer.size());

        luisa::vector<int32> result(array_size);
        stream << value_out_buffer.copy_to(result.data()) << synchronize();
        luisa::vector<int32> expected(array_size);
        std::exclusive_scan(input_data.begin(), input_data.end(), expected.begin(), 0);

        for(auto i = 0; i < segments; i++)
        {
            luisa::vector<int32> expect_segment(items_per_segment, 0);
            std::exclusive_scan(input_data.begin() + i * items_per_segment,
                                input_data.begin()
                                    + std::min((i + 1) * items_per_segment, static_cast<int>(array_size)),
                                expect_segment.begin(),
                                0);
            for(auto j = i * items_per_segment; j < (i + 1) * items_per_segment && j < array_size; j++)
            {
                LUISA_INFO("index:{} Key: {}, Expected Aggregate: {}, result:{}",
                           j,
                           i,
                           expect_segment[j - i * items_per_segment],
                           result[j]);
                expect(expect_segment[j - i * items_per_segment] == result[j]);
            }
        }
    };
}
