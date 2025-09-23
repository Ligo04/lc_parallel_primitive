// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */

#include "luisa/core/logging.h"
#include "luisa/runtime/buffer.h"
#include "luisa/vstl/config.h"
#include <lc_parallel_primitive/parallel_primitive.h>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

int main(int argc, char* argv[])
{

    log_level_verbose();

    Context context{argv[1]};
    Device  device = context.create_device("cuda");

    DeviceReduce reducer;
    reducer.create(device);

    auto in_buffer   = device.create_buffer<int32>(1024);
    auto temp_buffer = device.create_buffer<int32>(3);
    auto out_buffer  = device.create_buffer<int32>(1);

    std::vector<int32> input_data(1024);
    for(int i = 0; i < 1024; i++)
    {
        input_data[i] = i;
    }

    Stream stream = device.create_stream();
    stream << in_buffer.copy_from(input_data.data()) << synchronize();
    CommandList cmdlist;
    reducer.reduce(cmdlist,
                   temp_buffer.view(),
                   in_buffer.view(),
                   out_buffer.view(),
                   in_buffer.size(),
                   0);
    stream << cmdlist.commit() << synchronize();

    // 读取结果
    std::vector<int32> result(1);
    stream << out_buffer.copy_to(result.data()) << synchronize();  // 输出结果
    LUISA_INFO("Reduce: {}", result[0]);
    LUISA_INFO("Result (0+1+2+...+1023): {}", (1023 * 1024) / 2);
    return 0;
}