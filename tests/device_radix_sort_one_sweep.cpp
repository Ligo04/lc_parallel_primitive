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
#include <lcpp/parallel_primitive.h>
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

    constexpr int32_t BLOCK_SIZE       = 256;
    constexpr int32_t ITEMS_PER_THREAD = 2;
    constexpr int32_t WARP_NUMS        = 32;

    DeviceRadixSort<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD> radixsorter;
    radixsorter.create(device);

    "segment_reduce"_test = [&]
    {
        constexpr int32_t   array_size = 1024;
        luisa::vector<uint> input_data(array_size);
        for(int i = 0; i < array_size; i++)
        {
            input_data[i] = i;
        }
        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);

        auto key_buffer = device.create_buffer<uint>(array_size);
        stream << key_buffer.copy_from(input_data.data()) << synchronize();

        auto key_out_buffer = device.create_buffer<uint>(array_size);
        radixsorter.SortKeys<uint>(
            cmdlist, stream, key_buffer.view(), key_out_buffer.view(), key_buffer.size());
    };

    return 0;
}