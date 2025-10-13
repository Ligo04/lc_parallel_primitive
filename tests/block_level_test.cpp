
#include "lc_parallel_primitive/block/block_reduce.h"
#include "lc_parallel_primitive/block/block_scan.h"
#include "lc_parallel_primitive/runtime/core.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include <cstddef>
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


    constexpr size_t BLOCKSIZE        = 256;
    constexpr size_t array_size       = 10240;
    constexpr size_t ITEMS_PER_THREAD = 4;

    constexpr size_t ITEM_BLOCK_SIZE = 128;

    auto in_buffer  = device.create_buffer<int32>(array_size);
    auto out_buffer = device.create_buffer<int32>(array_size / BLOCKSIZE);
    std::vector<int32> result(array_size / BLOCKSIZE);

    std::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
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

    stream << (*block_reduce_shader)(in_buffer.view(), out_buffer.view(), array_size)
                  .dispatch(array_size);
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果

    "test_block_reduce"_test = [&]
    {
        for(auto i = 0; i < array_size / BLOCKSIZE; ++i)
        {
            auto index_result = std::accumulate(input_data.begin() + i * BLOCKSIZE,
                                                input_data.begin() + (i + 1) * BLOCKSIZE,
                                                0);
            LUISA_INFO("index: {}, index_result: {}, block_reduce_result: {}",
                       i,
                       index_result,
                       result[i]);
            expect(result[i] == index_result);
        }
    };


    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> block_reduce_items_shader =
        nullptr;
    lazy_compile(device,
                 block_reduce_items_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(ITEM_BLOCK_SIZE);
                     UInt tid = UInt(thread_id().x);
                     UInt block_start =
                         block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);

                     ArrayVar<int, ITEMS_PER_THREAD> thread_data;
                     $for(i, 0u, UInt(ITEMS_PER_THREAD))
                     {
                         UInt index = block_start + tid * UInt(ITEMS_PER_THREAD) + i;
                         thread_data[i] = select(0, arr_in.read(index), index < n);
                     };
                     Int aggregate =
                         BlockReduce<int, ITEM_BLOCK_SIZE, ITEMS_PER_THREAD>().Sum(thread_data);
                     $if(thread_id().x == 0)
                     {
                         arr_out.write(block_id().x, aggregate);
                     };
                 });

    stream << (*block_reduce_items_shader)(in_buffer.view(), out_buffer.view(), array_size)
                  .dispatch(array_size / ITEMS_PER_THREAD);
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果

    "test_block_reduce_4"_test = [&]
    {
        for(auto i = 0; i < array_size / (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD); ++i)
        {
            auto index_result = std::accumulate(
                input_data.begin() + i * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD),
                input_data.begin() + (i + 1) * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD),
                0);
            LUISA_INFO("index: {}, index_result: {}, block_reduce_result: {}",
                       i,
                       index_result,
                       result[i]);
            expect(result[i] == index_result);
        }
    };


    stream << in_buffer.copy_from(input_data.data()) << synchronize();
    auto scan_out_buffer = device.create_buffer<int32>(array_size);
    std::vector<int32> scan_result(array_size);
    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> block_scan_shader = nullptr;
    lazy_compile(device,
                 block_scan_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(BLOCKSIZE);
                     Int thid = Int(dispatch_id().x);

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

    stream << (*block_scan_shader)(in_buffer.view(), scan_out_buffer.view(), array_size)
                  .dispatch(array_size);
    stream << scan_out_buffer.copy_to(scan_result.data()) << synchronize();  // 输出结果

    "test_exlusive_scan"_test = [&]
    {
        for(auto i = 0; i < array_size / BLOCKSIZE; ++i)
        {
            std::vector<int> exclusive_scan_result(BLOCKSIZE);
            std::exclusive_scan(input_data.begin() + i * BLOCKSIZE,
                                input_data.begin() + (i + 1) * BLOCKSIZE,
                                exclusive_scan_result.begin(),
                                0);

            for(auto j = 0; j < BLOCKSIZE; ++j)
            {
                LUISA_INFO("index: {}, exclusive_scan_result: {}, scan_result: {}",
                           i * BLOCKSIZE + j,
                           exclusive_scan_result[j],
                           scan_result[i * BLOCKSIZE + j]);
                expect(exclusive_scan_result[j] == scan_result[i * BLOCKSIZE + j]);
            }
        }
    };

    luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> block_scan_item_shader = nullptr;
    lazy_compile(device,
                 block_scan_item_shader,
                 [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                 {
                     luisa::compute::set_block_size(ITEM_BLOCK_SIZE);
                     UInt tid = UInt(thread_id().x);
                     UInt block_start =
                         block_id().x * block_size_x() * UInt(ITEMS_PER_THREAD);

                     ArrayVar<int, ITEMS_PER_THREAD> thread_data;
                     $for(i, 0u, UInt(ITEMS_PER_THREAD))
                     {
                         UInt index = block_start + tid * UInt(ITEMS_PER_THREAD) + i;
                         thread_data[i] = select(0, arr_in.read(index), index < n);
                     };

                     ArrayVar<int, ITEMS_PER_THREAD> scanned_data;
                     BlockScan<int, ITEM_BLOCK_SIZE, ITEMS_PER_THREAD>().ExclusiveSum(
                         thread_data, scanned_data);

                     $for(i, 0u, UInt(ITEMS_PER_THREAD))
                     {
                         UInt index = block_start + tid * UInt(ITEMS_PER_THREAD) + i;
                         arr_out.write(index, select(0, scanned_data[i], index < n));
                     };
                 });


    stream << (*block_scan_item_shader)(in_buffer.view(), scan_out_buffer.view(), array_size)
                  .dispatch(array_size / ITEMS_PER_THREAD);
    stream << scan_out_buffer.copy_to(scan_result.data()) << synchronize();  // 输出结果

    "test_exlusive_scan_4"_test = [&]
    {
        for(auto i = 0; i < array_size / (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD); ++i)
        {
            std::vector<int> exclusive_scan_result((ITEM_BLOCK_SIZE * ITEMS_PER_THREAD));
            std::exclusive_scan(input_data.begin() + i * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD),
                                input_data.begin() + (i + 1) * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD),
                                exclusive_scan_result.begin(),
                                0);

            for(auto j = 0; j < (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD); ++j)
            {
                LUISA_INFO("index: {}, exclusive_scan_result: {}, scan_result: {}",
                           i * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD) + j,
                           exclusive_scan_result[j],
                           scan_result[i * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD) + j]);
                expect(exclusive_scan_result[j]
                       == scan_result[i * (ITEM_BLOCK_SIZE * ITEMS_PER_THREAD) + j]);
            }
        }
    };

    std::cout << std::endl;
}