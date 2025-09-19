/*
 * @Author: Ligo 
 * @Date: 2025-09-19 16:04:31 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-19 17:09:25
 */

#include <lc_parallel_primitive/parallel_primitive.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

int main(int argc, char* argv[])
{

    log_level_info();

    Context context{argv[1]};
    Device  device = context.create_device("cuda");

    DeviceReduce reducer;
}
