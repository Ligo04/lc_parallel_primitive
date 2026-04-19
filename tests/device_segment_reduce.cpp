// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */


#include <cmath>
#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <numeric>
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
    backend_name = "cuda";
#elif __APPLE__
    backend_name = "metal";
#else
    backend_name = "cuda";
#endif

    // Parse command-line arguments: --ctx=./dir --backend=dx
    for(int i = 1; i < argc; ++i)
    {
        luisa::string_view arg(argv[i]);
        if(arg.starts_with("--ctx="))
        {
            ctx_dir = arg.substr(6);
            LUISA_INFO("Use context directory from command-line: {}", ctx_dir);
        }
        else if(arg.starts_with("--backend="))
        {
            backend_name = arg.substr(10);
            LUISA_INFO("Use backend from command-line: {}", backend_name);
        }
    }

    Context     context{ctx_dir.empty() ? argv[0] : ctx_dir};
    Device      device = context.create_device(backend_name);
    Stream      stream = device.create_stream();
    CommandList cmdlist;

    DeviceSegmentReduce<> reducer;
    reducer.create(device, &stream);

    "segment_reduce"_test = [&]
    {
        constexpr int32_t    array_size = 1024;
        luisa::vector<int32> input_data(array_size);
        for(int i = 0; i < array_size; i++)
        {
            input_data[i] = i;
        }
        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);

        constexpr int       items_per_segment = 100;
        auto                in_buffer         = device.create_buffer<int32>(array_size);
        auto                num_segments      = array_size / items_per_segment + 1;
        auto                out_buffer        = device.create_buffer<int32>(num_segments);
        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment] = std::min(i + items_per_segment, array_size);
            }
        }

        auto begin_offsets = device.create_buffer<uint>(num_segments);
        auto end_offsets   = device.create_buffer<uint>(num_segments);

        stream << in_buffer.copy_from(input_data.data()) << synchronize();
        stream << begin_offsets.copy_from(begin_offsets_array.data()) << synchronize();
        stream << end_offsets.copy_from(end_offsets_array.data()) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceSegmentReduce<>::GetTempStorageBytes<int32>(array_size);
        auto temp_buffer  = device.create_buffer<uint>(std::max<size_t>(1, bytes_to_uint_count(temp_bytes)));

        reducer.Sum(cmdlist,
                    temp_buffer.view(),
                    in_buffer.view(),
                    out_buffer.view(),
                    num_segments,
                    begin_offsets.view(),
                    end_offsets.view());
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_sum =
                    std::accumulate(input_data.begin() + i,
                                    input_data.begin() + i + std::min(array_size - i, items_per_segment),
                                    0);
                LUISA_INFO("Segment {}: expected sum = {}, got {}",
                           i / items_per_segment,
                           expected_sum,
                           result[i / items_per_segment]);
                expect(expected_sum == result[i / items_per_segment]);
            }
        }
    };

    "fixed_segment_reduce"_test = [&]
    {
        constexpr int32_t fixed_array       = 1024;
        constexpr int     items_per_segment = 32;
        static_assert(fixed_array % items_per_segment == 0, "fixed_array must be divisible by items_per_segment");
        auto num_segments = fixed_array / items_per_segment;

        auto                in_buffer  = device.create_buffer<int32>(fixed_array);
        auto                out_buffer = device.create_buffer<int32>(num_segments);
        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment]   = i + items_per_segment;
            }
        }

        luisa::vector<int32> input_data(fixed_array);
        for(int i = 0; i < fixed_array; i++)
        {
            input_data[i] = i;
        }
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceSegmentReduce<>::GetTempStorageBytes<int32>(fixed_array);
        auto temp_buffer  = device.create_buffer<uint>(std::max<size_t>(1, bytes_to_uint_count(temp_bytes)));

        reducer.Sum(cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), num_segments, items_per_segment);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_sum =
                    std::accumulate(input_data.begin() + i, input_data.begin() + i + items_per_segment, 0);
                LUISA_INFO("Segment {}: expected sum = {}, got {}",
                           i / items_per_segment,
                           expected_sum,
                           result[i / items_per_segment]);
                expect(expected_sum == result[i / items_per_segment]);
            }
        }
    };


    "segment_reduce_arg_max"_test = [&]
    {
        constexpr int32_t array_size        = 1024;
        constexpr int     items_per_segment = 100;


        luisa::vector<int32> input_data(array_size);
        for(int i = 0; i < array_size; i++)
        {
            input_data[i] = i;
        }
        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);

        auto in_buffer    = device.create_buffer<int32>(array_size);
        auto num_segments = ceil_div(array_size, items_per_segment);

        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment] = std::min(i + items_per_segment, array_size);
            }
        }

        auto begin_offsets = device.create_buffer<uint>(num_segments);
        auto end_offsets   = device.create_buffer<uint>(num_segments);

        stream << in_buffer.copy_from(input_data.data()) << synchronize();
        stream << begin_offsets.copy_from(begin_offsets_array.data()) << synchronize();
        stream << end_offsets.copy_from(end_offsets_array.data()) << synchronize();

        auto out_buffer       = device.create_buffer<int>(num_segments);
        auto index_out_buffer = device.create_buffer<uint>(num_segments);

        // CUB-style: get temp storage size for ArgMax
        size_t temp_bytes = DeviceSegmentReduce<>::GetArgTempStorageBytes<int32>(array_size, num_segments);
        auto temp_buffer  = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        reducer.ArgMax(cmdlist,
                       temp_buffer.view(),
                       in_buffer.view(),
                       out_buffer.view(),
                       index_out_buffer.view(),
                       num_segments,
                       begin_offsets.view(),
                       end_offsets.view());
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_max =
                    std::max_element(input_data.begin() + i,
                                     input_data.begin() + i + std::min(array_size - i, items_per_segment));
                LUISA_INFO("Segment {}: expected max = {}, got {}",
                           i / items_per_segment,
                           *expected_max,
                           result[i / items_per_segment]);
                expect(*expected_max == result[i / items_per_segment]);
            }
        }
    };


    "fixed_segment_reduce_arg_min"_test = [&]
    {
        constexpr int32_t fixed_array       = 1024;
        constexpr int     items_per_segment = 32;
        static_assert(fixed_array % items_per_segment == 0, "fixed_array must be divisible by items_per_segment");
        auto num_segments = fixed_array / items_per_segment;

        auto in_buffer        = device.create_buffer<int32>(fixed_array);
        auto out_buffer       = device.create_buffer<int32>(num_segments);
        auto index_out_buffer = device.create_buffer<uint>(num_segments);

        luisa::vector<int32> input_data(fixed_array);
        for(int i = 0; i < fixed_array; i++)
        {
            input_data[i] = i;
        }

        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        // CUB-style: get temp storage size for ArgMin
        size_t temp_bytes = DeviceSegmentReduce<>::GetArgTempStorageBytes<int32>(fixed_array, num_segments);
        auto temp_buffer  = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        reducer.ArgMin(cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), index_out_buffer.view(), num_segments, items_per_segment);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(result.data()) << synchronize();
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_min =
                    std::min_element(input_data.begin() + i, input_data.begin() + i + items_per_segment);
                LUISA_INFO("Segment {}: expected arg min = {}, got {}",
                           i / items_per_segment,
                           *expected_min,
                           result[i / items_per_segment]);
                expect(*expected_min == result[i / items_per_segment]);
            }
        }
    };
}
