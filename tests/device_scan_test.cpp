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

    constexpr int32_t BLOCK_SIZE       = 256;
    constexpr int32_t ITEMS_PER_THREAD = 4;
    constexpr int32_t WARP_NUMS        = 32;
    constexpr int32_t MAX_NUM_LOGIC    = 24;

    using ScannerT = DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
    ScannerT scanner;
    scanner.create(device, &stream);

    "exclusive_scan"_test = [&]
    {
        for(int i = 0; i < MAX_NUM_LOGIC; i++)
        {
            const uint          array_size = 1 << i;
            luisa::vector<uint> input_data(array_size, 1);
            auto                in_buffer            = device.create_buffer<uint>(array_size);
            auto                exclusive_out_buffer = device.create_buffer<uint>(array_size);
            stream << in_buffer.copy_from(input_data.data()) << synchronize();

            // CUB-style: get temp storage size
            size_t temp_bytes = ScannerT::GetTempStorageBytes<uint>(array_size);
            auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

            scanner.ExclusiveSum(
                cmdlist, temp_buffer.view(), in_buffer.view(), exclusive_out_buffer.view(), in_buffer.size());
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> exclusive_result(array_size);
            stream << exclusive_out_buffer.copy_to(exclusive_result.data()) << synchronize();

            auto inclusive_out_buffer = device.create_buffer<uint>(array_size);

            // Reuse same temp_buffer for inclusive scan (same size requirement)
            scanner.InclusiveSum(
                cmdlist, temp_buffer.view(), in_buffer.view(), inclusive_out_buffer.view(), in_buffer.size());
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> inclusive_result(array_size);
            stream << inclusive_out_buffer.copy_to(inclusive_result.data()) << synchronize();

            luisa::vector<uint> exclusive_expected(array_size);
            luisa::vector<uint> inclusive_expected(array_size);
            std::exclusive_scan(input_data.begin(), input_data.end(), exclusive_expected.begin(), 0);
            std::inclusive_scan(input_data.begin(), input_data.end(), inclusive_expected.begin());

            auto exclusive_expected_match =
                std::equal(exclusive_result.begin(), exclusive_result.end(), exclusive_expected.begin());
            auto inclusive_expected_match =
                std::equal(inclusive_result.begin(), inclusive_result.end(), inclusive_expected.begin());
            expect(exclusive_expected_match && inclusive_expected_match)
                << "Scan failed for array size " << i << "\n";
        }
    };


    "exclusive_scan_by_key"_test = [&]
    {
        for(auto i = 5; i < MAX_NUM_LOGIC; i++)
        {
            const uint           array_size = 1 << i;
            luisa::vector<int32> input_data(array_size, 1);
            auto                 key_buffer   = device.create_buffer<int32>(array_size);
            auto                 value_buffer = device.create_buffer<int32>(array_size);

            constexpr int items_per_segment = 100;
            const int     segments = (array_size + items_per_segment - 1) / items_per_segment;

            luisa::vector<int32> input_keys(array_size);
            for(auto i = 0; i < array_size; i++)
            {
                input_keys[i] = i / items_per_segment;
            }

            LUISA_INFO("Array size: {}, Items per segment: {}, Total segments: {}", array_size, items_per_segment, segments);

            stream << key_buffer.copy_from(input_keys.data()) << synchronize();
            stream << value_buffer.copy_from(input_data.data()) << synchronize();

            auto value_out_buffer = device.create_buffer<int32>(array_size);

            // CUB-style: get temp storage size for ScanByKey
            size_t temp_bytes = ScannerT::GetScanByKeyTempStorageBytes<int32, int32>(array_size);
            auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

            scanner.ExclusiveSumByKey(cmdlist,
                                      temp_buffer.view(),
                                      key_buffer.view(),
                                      value_buffer.view(),
                                      value_out_buffer.view(),
                                      key_buffer.size());
            stream << cmdlist.commit() << synchronize();

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
                    expect(expect_segment[j - i * items_per_segment] == result[j]);
                }
            }
        }
    };
}
