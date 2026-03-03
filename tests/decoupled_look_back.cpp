
#include "lcpp/device/details/single_pass_scan_operator.h"
#include "lcpp/runtime/core.h"
#include "luisa/core/logging.h"
#include "luisa/core/stl/vector.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include "luisa/runtime/stream.h"
#include <cstddef>
#include <lcpp/parallel_primitive.h>
#include <boost/ut.hpp>
#include <numeric>
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
    CommandList cmdlist;
    Stream      stream = device.create_stream();

    constexpr size_t WARP_SIZE  = 32;
    constexpr size_t BLOCK_SIZE = 256;
    constexpr size_t NUM_TILES  = 102400;
    const size_t     num_blocks = ceil_div(NUM_TILES, BLOCK_SIZE);

    "decoupled_look_back_int"_test = [&]()
    {
        auto scan_tile_status_buffer          = device.create_buffer<uint>(WARP_SIZE + NUM_TILES);
        auto scan_tile_value_partial_buffer   = device.create_buffer<int>(WARP_SIZE + NUM_TILES);
        auto scan_tile_value_inclusive_buffer = device.create_buffer<int>(WARP_SIZE + NUM_TILES);

        luisa::unique_ptr<Shader<1, Buffer<uint>, uint>> init_kernel = nullptr;
        lazy_compile(device,
                     init_kernel,
                     [&](BufferVar<uint> tile_status, UInt num_tiles) noexcept
                     { InitializeWardStatus(num_tiles, tile_status); });

        cmdlist << (*init_kernel)(scan_tile_status_buffer.view(), NUM_TILES).dispatch(num_blocks * BLOCK_SIZE);
        stream << cmdlist.commit() << synchronize();

        auto scan_op = [](const Var<int>& a, const Var<int>& b) noexcept { return a + b; };

        luisa::unique_ptr<Shader<1, Buffer<uint>, Buffer<int>, Buffer<int>, Buffer<int>, Buffer<int>>> decoupled_look_back_kernel =
            nullptr;
        lazy_compile(device,
                     decoupled_look_back_kernel,
                     [&](BufferVar<uint> tile_status,
                         BufferVar<int>  tile_partial,
                         BufferVar<int>  tile_inclusive,
                         BufferVar<int>  exclusive_output,
                         BufferVar<int>  inclusive_output) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         compute::UInt tid = compute::thread_x();

                         ScanTileStateViewer<int> viewer{tile_status, tile_partial, tile_inclusive};

                         using tile_prefix_op =
                             TilePrefixCallbackOp<int, decltype(scan_op), ScanTileStateViewer<int>, no_delay_constructor<int>>;

                         auto temp_storage = luisa::compute::Shared<TilePrefixTempStorage<int>>(1);

                         tile_prefix_op prefix(viewer, &temp_storage, scan_op);
                         const auto     tile_idx        = prefix.GetTileIndex();
                         compute::Int   block_aggregate = 1;
                         $if(tile_idx == 0)
                         {
                             $if(tid == 0)
                             {
                                 viewer.SetInclusive(tile_idx, block_aggregate);
                                 exclusive_output.write(tile_idx, 0);
                                 inclusive_output.write(tile_idx, block_aggregate);
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
                                     Var<int> inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
                                     exclusive_output.write(tile_idx, exclusive_prefix);
                                     inclusive_output.write(tile_idx, inclusive_prefix);
                                 };
                             };
                         };
                     });

        auto exclusive_buffer = device.create_buffer<int>(NUM_TILES);
        auto inclusive_buffer = device.create_buffer<int>(NUM_TILES);
        cmdlist << (*decoupled_look_back_kernel)(scan_tile_status_buffer.view(),
                                                 scan_tile_value_partial_buffer.view(),
                                                 scan_tile_value_inclusive_buffer.view(),
                                                 exclusive_buffer.view(),
                                                 inclusive_buffer.view())
                       .dispatch(NUM_TILES * BLOCK_SIZE);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int> exclusive_result(NUM_TILES);
        luisa::vector<int> inclusive_result(NUM_TILES);
        stream << exclusive_buffer.copy_to(exclusive_result.data())
               << inclusive_buffer.copy_to(inclusive_result.data()) << synchronize();

        luisa::vector<int> data(NUM_TILES);
        for(auto i = 0; i < NUM_TILES; i++)
        {
            data[i] = 1;
        }
        luisa::vector<int> exclusive_expected(NUM_TILES);
        luisa::vector<int> inclusive_expected(NUM_TILES);
        std::exclusive_scan(data.begin(), data.end(), exclusive_expected.begin(), 0);
        std::inclusive_scan(data.begin(), data.end(), inclusive_expected.begin());

        bool exclusive_pass =
            std::equal(exclusive_result.begin(), exclusive_result.end(), exclusive_expected.begin());
        // expect(exclusive_pass) << "Decoupled look-back exclusive scan failed.";
        bool inclusive_pass =
            std::equal(inclusive_result.begin(), inclusive_result.end(), inclusive_expected.begin());
        // expect(inclusive_pass) << "Decoupled look-back inclusive scan failed.";
        expect(exclusive_pass && inclusive_pass);
        if(!exclusive_pass || !inclusive_pass)
        {
            for(size_t i = 0; i < NUM_TILES; i++)
            {
                if(exclusive_result[i] != exclusive_expected[i] || inclusive_result[i] != inclusive_expected[i])
                {
                    LUISA_INFO("Tile {}: exclusive_result = {}, inclusive_result = {} (expected: {}, {})",
                               i,
                               exclusive_result[i],
                               inclusive_result[i],
                               exclusive_expected[i],
                               inclusive_expected[i]);
                }
            }
        }
    };

    "decoupled_look_back_key_value_pair"_test = [&]()
    {
        using KVP = luisa::parallel_primitive::KeyValuePair<int, int>;

        auto sum_op = [](const Var<int>& a, const Var<int>& b) noexcept { return a + b; };

        using KVPScanOp                       = ReduceByKeyOp<decltype(sum_op)>;
        auto scan_tile_status_buffer          = device.create_buffer<uint>(WARP_SIZE + NUM_TILES);
        auto scan_tile_value_partial_buffer   = device.create_buffer<KVP>(WARP_SIZE + NUM_TILES);
        auto scan_tile_value_inclusive_buffer = device.create_buffer<KVP>(WARP_SIZE + NUM_TILES);

        luisa::unique_ptr<Shader<1, Buffer<uint>, uint>> init_kernel = nullptr;
        lazy_compile(device,
                     init_kernel,
                     [&](BufferVar<uint> tile_status, UInt num_tiles) noexcept
                     { InitializeWardStatus(num_tiles, tile_status); });

        cmdlist << (*init_kernel)(scan_tile_status_buffer.view(), NUM_TILES).dispatch(num_blocks * BLOCK_SIZE);
        stream << cmdlist.commit() << synchronize();


        luisa::unique_ptr<Shader<1, Buffer<uint>, Buffer<KVP>, Buffer<KVP>, Buffer<KVP>, Buffer<KVP>>> decoupled_look_back_kernel =
            nullptr;
        lazy_compile(device,
                     decoupled_look_back_kernel,
                     [&](BufferVar<uint> tile_status,
                         BufferVar<KVP>  tile_partial,
                         BufferVar<KVP>  tile_inclusive,
                         BufferVar<KVP>  tile_exclusive_output,
                         BufferVar<KVP>  tile_inclusive_output) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         compute::UInt            tid = compute::thread_x();
                         ScanTileStateViewer<KVP> viewer{tile_status, tile_partial, tile_inclusive};


                         using tile_prefix_op =
                             TilePrefixCallbackOp<KVP, KVPScanOp, ScanTileStateViewer<KVP>>;

                         auto temp_storage = luisa::compute::Shared<TilePrefixTempStorage<KVP>>(1);
                         KVPScanOp         scan_op{sum_op};
                         tile_prefix_op    prefix(viewer, &temp_storage, scan_op);
                         const auto        tile_idx = prefix.GetTileIndex();
                         compute::Var<KVP> block_aggregate;
                         block_aggregate.key   = def(1);
                         block_aggregate.value = 1;
                         $if(tile_idx == 0)
                         {
                             $if(tid == 0)
                             {
                                 viewer.SetInclusive(tile_idx, block_aggregate);
                                 Var<KVP> zero;
                                 zero.key   = 0;
                                 zero.value = 0;
                                 tile_exclusive_output.write(tile_idx, zero);
                                 tile_inclusive_output.write(tile_idx, block_aggregate);
                             };
                         }
                         $else
                         {
                             const auto warp_id = tid / luisa::compute::UInt(WARP_SIZE);

                             $if(warp_id == 0)
                             {
                                 Var<KVP> exclusive_prefix = prefix(block_aggregate);
                                 $if(tid == 0)
                                 {
                                     Var<KVP> inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
                                     tile_exclusive_output.write(tile_idx, exclusive_prefix);
                                     tile_inclusive_output.write(tile_idx, inclusive_prefix);
                                 };
                             };
                         };
                     });

        auto exclusive_buffer = device.create_buffer<KVP>(NUM_TILES);
        auto inclusive_buffer = device.create_buffer<KVP>(NUM_TILES);
        cmdlist << (*decoupled_look_back_kernel)(scan_tile_status_buffer.view(),
                                                 scan_tile_value_partial_buffer.view(),
                                                 scan_tile_value_inclusive_buffer.view(),
                                                 exclusive_buffer.view(),
                                                 inclusive_buffer.view())
                       .dispatch(NUM_TILES * BLOCK_SIZE);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<KVP> exclusive_result(NUM_TILES);
        luisa::vector<KVP> inclusive_result(NUM_TILES);
        stream << exclusive_buffer.copy_to(exclusive_result.data())
               << inclusive_buffer.copy_to(inclusive_result.data()) << synchronize();
        luisa::vector<int> data(NUM_TILES);
        for(auto i = 0; i < NUM_TILES; i++)
        {
            data[i] = 1;
        }
        luisa::vector<int> exclusive_expected(NUM_TILES);
        luisa::vector<int> inclusive_expected(NUM_TILES);
        std::exclusive_scan(data.begin(), data.end(), exclusive_expected.begin(), 0);
        std::inclusive_scan(data.begin(), data.end(), inclusive_expected.begin());

        bool exclusive_pass = std::equal(exclusive_result.begin(),
                                         exclusive_result.end(),
                                         exclusive_expected.begin(),
                                         [](const KVP& a, const int& b) { return a.value == b; });
        bool inclusive_pass = std::equal(inclusive_result.begin(),
                                         inclusive_result.end(),
                                         inclusive_expected.begin(),
                                         [](const KVP& a, const int& b) { return a.value == b; });
        expect(exclusive_pass && inclusive_pass);
        if(!exclusive_pass || !inclusive_pass)
        {
            for(size_t i = 0; i < NUM_TILES; i++)
            {
                if(exclusive_result[i].value != exclusive_expected[i]
                   || inclusive_result[i].value != inclusive_expected[i])
                {
                    LUISA_INFO("Tile {}: exclusive_result = {}, inclusive_result = {} (expected: {}, {})",
                               i,
                               exclusive_result[i],
                               inclusive_result[i],
                               exclusive_expected[i],
                               inclusive_expected[i]);
                }
            }
        }
    };
}