
#include "lcpp/block/block_reduce.h"
#include "lcpp/block/block_scan.h"
#include "lcpp/common/keyvaluepair.h"
#include "lcpp/device/details/single_pass_scan_operator.h"
#include "lcpp/runtime/core.h"
#include "lcpp/warp/warp_reduce.h"
#include "lcpp/warp/warp_scan.h"
#include "luisa/core/basic_traits.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/func.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include "luisa/runtime/stream.h"
#include "luisa/vstl/config.h"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <boost/ut.hpp>
#include <limits>
#include <numeric>
#include <vector>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using namespace boost::ut;
;

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

    constexpr size_t WARP_SIZE  = 32;
    constexpr size_t array_size = 256;
    constexpr size_t BLOCK_SIZE = 256;
    constexpr size_t NUM_TILES  = 10000;
    const size_t     num_blocks = ceil(float(NUM_TILES) / BLOCK_SIZE);
    auto scan_tile_buffer = device.create_buffer<ScanTileState<int>>(WARP_SIZE + NUM_TILES);


    auto exclusive_buffer = device.create_buffer<int>(NUM_TILES);
    auto inclusive_buffer = device.create_buffer<int>(NUM_TILES);

    luisa::unique_ptr<Shader<1, Buffer<ScanTileState<int>>>> init_kernel = nullptr;
    lazy_compile(device,
                 init_kernel,
                 [&](BufferVar<ScanTileState<int>> tile_state) noexcept
                 {
                     ScanTileStateViewer<int>().InitializeWardStatus(tile_state, NUM_TILES);
                 });

    cmdlist << (*init_kernel)(scan_tile_buffer.view()).dispatch(num_blocks);
    stream << cmdlist.commit() << synchronize();

    auto scan_op = [](const Var<int>& a, const Var<int>& b) noexcept
    { return a + b; };


    luisa::unique_ptr<Shader<1, Buffer<ScanTileState<int>>, Buffer<int>, Buffer<int>>> decoupled_look_back_kernel =
        nullptr;
    lazy_compile(
        device,
        decoupled_look_back_kernel,
        [&](BufferVar<ScanTileState<int>> tile_state,
            BufferVar<int>                exclusive_output,
            BufferVar<int>                inclusive_output) noexcept
        {
            luisa::compute::set_block_size(BLOCK_SIZE);
            luisa::compute::set_warp_size(WARP_SIZE);
            compute::UInt tid = compute::thread_x();
            using tile_prefix_op =
                TilePrefixCallbackOp<int, decltype(scan_op), ScanTileState<int>>;

            auto temp_storage =
                new luisa::compute::Shared<TilePrefixTempStorage<int>>{1};

            tile_prefix_op prefix(tile_state, temp_storage, scan_op);
            const auto     tile_idx        = prefix.GetTileIndex();
            compute::Int   block_aggregate = block_id().x;
            $if(tile_idx == 0)
            {
                $if(tid == 0)
                {
                    ScanTileStateViewer<int>().SetInclusive(tile_state, tile_idx, block_aggregate);
                    exclusive_output.write(tile_idx, 0);
                    inclusive_output.write(tile_idx, 0);
                };
            }
            $else
            {
                const auto warp_id = tid / luisa::compute::UInt(WARP_SIZE);

                $if(warp_id == 0)
                {
                    Var<int> exclusive_prefix = prefix(block_aggregate);
                    $if(tid == 0)
                    {
                        Var<int> inclusive_prefix =
                            scan_op(exclusive_prefix, block_aggregate);
                        exclusive_output.write(tile_idx, exclusive_prefix);
                        inclusive_output.write(tile_idx, inclusive_prefix);
                    };
                };
            };
        });

    cmdlist << (*decoupled_look_back_kernel)(scan_tile_buffer.view(),
                                             exclusive_buffer.view(),
                                             inclusive_buffer.view())
                   .dispatch(NUM_TILES * BLOCK_SIZE);
    stream << cmdlist.commit() << synchronize();

    luisa::vector<int> exclusive_result(NUM_TILES);
    luisa::vector<int> inclusive_result(NUM_TILES);
    stream << exclusive_buffer.copy_to(exclusive_result.data())
           << inclusive_buffer.copy_to(inclusive_result.data()) << synchronize();

    for(size_t i = 0; i < NUM_TILES; i++)
    {
        LUISA_INFO("Tile {}: exclusive_result = {}, inclusive_result = {}",
                   i,
                   exclusive_result[i],
                   inclusive_result[i]);
    }
};