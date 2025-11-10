
#include "lcpp/block/block_reduce.h"
#include "lcpp/block/block_scan.h"
#include "lcpp/runtime/core.h"
#include "lcpp/warp/warp_reduce.h"
#include "lcpp/warp/warp_scan.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include "luisa/vstl/config.h"
#include <cmath>
#include <cstddef>
#include <lcpp/parallel_primitive.h>
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
    Device device = context.create_device("dx");
#elif __APPLE__
    Device device = context.create_device("metal");
#else
    Device device = context.create_device("cuda");
#endif
    CommandList cmdlist;
    Stream      stream = device.create_stream();


    constexpr size_t WARP_SIZE  = 32;
    constexpr size_t array_size = 512;
    constexpr size_t BLOCK_SIZE = 256;

    auto in_buffer = device.create_buffer<int32>(array_size);
    auto reduce_out_buffer = device.create_buffer<int32>(array_size / WARP_SIZE);

    luisa::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }

    "test_warp_reduce"_test = [&]
    {
        luisa::vector<int32> result(array_size / WARP_SIZE);
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_reduce_test_shader =
            nullptr;
        lazy_compile(device,
                     warp_reduce_test_shader,
                     [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         Int thid =
                             Int(block_size().x * block_id().x + thread_id().x);
                         Int thread_data = def(0);
                         $if(thid < n)
                         {
                             thread_data = arr_in.read(thid);
                         };
                         Int result = WarpReduce<int>().Reduce(
                             thread_data,
                             [](const Var<int>& a, const Var<int>& b) noexcept
                             { return a + b; });
                         $if(compute::warp_lane_id() == 0)
                         {
                             arr_out.write(thid / UInt(WARP_SIZE), result);
                         };
                     });

        stream << (*warp_reduce_test_shader)(in_buffer.view(), reduce_out_buffer.view(), array_size)
                      .dispatch(array_size);
        stream << reduce_out_buffer.copy_to(result.data()) << synchronize();  // 输出结果

        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            auto index_result = std::accumulate(input_data.begin() + i * WARP_SIZE,
                                                input_data.begin() + (i + 1) * WARP_SIZE,
                                                0);
            LUISA_INFO("index: {}, index_result: {}, warp_reduce_result: {}",
                       i,
                       index_result,
                       result[i]);
            expect(result[i] == index_result);
        }
    };


    "test_warp_segment_reduce"_test = [&]
    {
        luisa::vector<int32> input_keys(array_size);
        constexpr int        ITEMS_PER_SEGMENT = 10;
        for(auto i = 0; i < array_size; i += ITEMS_PER_SEGMENT)
        {
            for(auto j = 0; j < ITEMS_PER_SEGMENT - 1 && i + j < array_size; j++)
            {
                input_keys[i + j] = 0;
            }
            if(i + ITEMS_PER_SEGMENT - 1 < array_size)
            {
                input_keys[i + ITEMS_PER_SEGMENT - 1] = 1;
            }
            // input_keys[i] = 1;
            // for(auto j = 1; j < ITEMS_PER_SEGMENT && i + j < array_size; j++)
            // {
            //     input_keys[i + j] = 0;
            // }
        }

        auto seg_result_num =
            std::ceil(array_size / ITEMS_PER_SEGMENT + array_size / WARP_SIZE
                      - (array_size * 2) / (ITEMS_PER_SEGMENT * WARP_SIZE));
        LUISA_INFO("Expected segments: {}", seg_result_num);

        auto key_buffer = device.create_buffer<int32>(array_size);
        auto segment_reduce_out_buffer = device.create_buffer<int32>(seg_result_num);

        stream << key_buffer.copy_from(input_keys.data()) << synchronize();
        stream << in_buffer.copy_from(input_data.data()) << synchronize();

        luisa::vector<int32> seg_result(seg_result_num);

        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, Buffer<int>, int>> warp_segment_reduce_shader =
            nullptr;
        lazy_compile(
            device,
            warp_segment_reduce_shader,
            [&](BufferVar<int> arr_in, BufferVar<int> key_in, BufferVar<int> arr_out, Int n) noexcept
            {
                luisa::compute::set_block_size(BLOCK_SIZE);
                luisa::compute::set_warp_size(WARP_SIZE);
                Int thid = Int(block_size().x * block_id().x + thread_id().x);
                Int thread_data = def(0);
                $if(thid < n)
                {
                    thread_data = arr_in.read(thid);
                };
                Int key_data = def(0);
                $if(thid < n)
                {
                    key_data = key_in.read(thid);
                };
                Int result = WarpReduce<int>().TailSegmentedSum(thread_data, key_data);
                $if(key_data == 1 | warp_lane_id() == 0)
                {
                    UInt index = (thid + 1) / UInt(ITEMS_PER_SEGMENT)
                                 + (thid + 1) / UInt(WARP_SIZE)
                                 - ((thid + 1) * 2) / UInt(ITEMS_PER_SEGMENT * WARP_SIZE);
                    arr_out.write(index, result);
                };
            });

        stream << (*warp_segment_reduce_shader)(in_buffer.view(),
                                                key_buffer.view(),
                                                segment_reduce_out_buffer.view(),
                                                array_size)
                      .dispatch(array_size);
        stream << segment_reduce_out_buffer.copy_to(seg_result.data())
               << synchronize();  // 输出结果

        for(auto i = 0; i < array_size;)
        {
            for(int j = 1; j < array_size; ++j)
            {
                auto curr_index = i + j;
                if(curr_index % ITEMS_PER_SEGMENT == 0 || curr_index % WARP_SIZE == 0)
                {
                    auto index = i / ITEMS_PER_SEGMENT + i / WARP_SIZE
                                 - (i * 2) / (ITEMS_PER_SEGMENT * WARP_SIZE);
                    auto index_result = std::accumulate(
                        input_data.begin() + i, input_data.begin() + curr_index, 0);
                    LUISA_INFO("segment start index: {}, start index: {}, index: {}, index_result: {}, warp_segment_reduce_result: {}",
                               i,
                               i,
                               index,
                               index_result,
                               seg_result[index]);
                    // expect(seg_result[index] == index_result);
                    i += j;
                    break;
                }
            }
        };
    };


    "test_warp_ex_scan"_test = [&]
    {
        auto scan_out_buffer = device.create_buffer<int32>(array_size);
        luisa::vector<int32> scan_result(array_size);
        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_ex_scan_test_shader =
            nullptr;
        lazy_compile(device,
                     warp_ex_scan_test_shader,
                     [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         Int thid =
                             Int(block_size().x * block_id().x + thread_id().x);
                         Int thread_data = def(0);
                         $if(thid < n)
                         {
                             thread_data = arr_in.read(thid);
                         };
                         Int output_block_scan;
                         Int warp_aggregate;
                         output_block_scan = warp_prefix_sum(thread_data);
                         WarpScan<int>().ExclusiveSum(thread_data, output_block_scan, warp_aggregate);
                         $if(thid < n)
                         {
                             arr_out.write(thid, output_block_scan);
                         };
                     });

        stream << (*warp_ex_scan_test_shader)(in_buffer.view(), scan_out_buffer.view(), array_size)
                      .dispatch(array_size);
        stream << scan_out_buffer.copy_to(scan_result.data()) << synchronize();  // 输出结果
        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            luisa::vector<int> exclusive_scan_result(WARP_SIZE);
            std::exclusive_scan(input_data.begin() + i * WARP_SIZE,
                                input_data.begin() + (i + 1) * WARP_SIZE,
                                exclusive_scan_result.begin(),
                                0);
            for(auto j = 0; j < WARP_SIZE; ++j)
            {
                LUISA_INFO("index: {}, index_result: {}, warp_exclusive_scan_result: {}",
                           i * WARP_SIZE + j,
                           exclusive_scan_result[j],
                           scan_result[i * WARP_SIZE + j]);
                expect(exclusive_scan_result[j] == scan_result[i * WARP_SIZE + j]);
            }
        };
    };
};