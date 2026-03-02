/*
 * @Author: Ligo
 * @Date: 2025-09-19 16:04:31
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-09-22 18:11:54
 */

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

    Context context{argv[0]};
    luisa::string_view backend_name;
#ifdef _WIN32
    backend_name = "cuda";
#elif __APPLE__
    backend_name = "metal";
#else
    backend_name = "cuda";
#endif
    if (argc > 1) {
        backend_name = argv[1];
        LUISA_INFO("Use {} backend from argv[1].", backend_name);
    } else {
        LUISA_INFO("Use {} backend defaultly.", backend_name);
    }
    Device device = context.create_device(backend_name);
    Stream      stream = device.create_stream();
    CommandList cmdlist;

    constexpr int32_t BLOCK_SIZE       = 128;
    constexpr int32_t ITEMS_PER_THREAD = 4;
    constexpr int32_t WARP_NUMS        = 32;

    DeviceRadixSort<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD> radixsorter;
    radixsorter.create(device);

    "radix sort key float"_test = [&]
    {
        for(uint loop = 5; loop < 25; ++loop)
        {
            uint array_size       = 1 << loop;
            using radix_sort_type = float;
            luisa::vector<radix_sort_type>                  input_key(array_size);
            luisa::vector<radix_sort_type>                  input_to_desc_key(array_size);
            std::mt19937                                    rng(114521);
            std::uniform_real_distribution<radix_sort_type> dist(0.0f, 1.0f);
            for(auto i = 0; i < array_size; i++)
            {
                input_key[i]         = dist(rng);
                input_to_desc_key[i] = dist(rng);
                // LUISA_INFO("Key {}: {} {}", i, input_key[i], input_to_desc_key[i]);
            }

            auto input_key_buffer = device.create_buffer<radix_sort_type>(array_size);
            stream << input_key_buffer.copy_from(input_key.data()) << synchronize();

            auto output_key_buffer = device.create_buffer<radix_sort_type>(array_size);
            radixsorter.SortKeys<radix_sort_type>(
                cmdlist, stream, input_key_buffer.view(), output_key_buffer.view(), input_key_buffer.size());

            luisa::vector<radix_sort_type> result(array_size);
            stream << output_key_buffer.copy_to(result.data()) << synchronize();

            input_key_buffer.release();
            output_key_buffer.release();

            // desc
            auto input_to_desc_key_buffer = device.create_buffer<radix_sort_type>(array_size);
            stream << input_to_desc_key_buffer.copy_from(input_to_desc_key.data()) << synchronize();

            auto out_desc_key_buffer = device.create_buffer<radix_sort_type>(array_size);
            radixsorter.SortKeysDescending<radix_sort_type>(cmdlist,
                                                            stream,
                                                            input_to_desc_key_buffer.view(),
                                                            out_desc_key_buffer.view(),
                                                            input_to_desc_key_buffer.size());

            luisa::vector<radix_sort_type> desc_result(array_size);
            stream << out_desc_key_buffer.copy_to(desc_result.data()) << synchronize();
            input_to_desc_key_buffer.release();
            out_desc_key_buffer.release();

            std::sort(input_key.begin(), input_key.end());
            std::sort(input_to_desc_key.begin(), input_to_desc_key.end(), std::greater<radix_sort_type>());

            bool pass = std::equal(result.begin(), result.end(), input_key.begin());
            bool desc_pass = std::equal(desc_result.begin(), desc_result.end(), input_to_desc_key.begin());
            expect(pass) << "Radix sort float key ascending failed at size " << array_size;
            expect(desc_pass) << "Radix sort float key descending failed at size " << array_size;
        }
    };

    "radix sort key uint variant size"_test = [&]
    {
        DeviceRadixSort<> device_radix_sort;
        device_radix_sort.create(device);
        for(uint loop = 0; loop < 24; ++loop)
        {
            uint                num_items  = 1 << loop;
            Buffer<uint>        d_keys_in  = device.create_buffer<uint>(num_items);
            Buffer<uint>        d_keys_out = device.create_buffer<uint>(num_items);
            luisa::vector<uint> host_keys(num_items);
            for(uint i = 0; i < num_items; ++i)
            {
                host_keys[i] = num_items - i - 1;
            }
            stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
            CommandList cmdlist;
            device_radix_sort.SortKeys(cmdlist, stream, d_keys_in.view(), d_keys_out.view(), num_items);
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> host_keys_out(num_items);
            stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

            std::sort(host_keys.begin(), host_keys.end());
            bool pass = std::equal(host_keys_out.begin(), host_keys_out.end(), host_keys.begin());
            expect(pass) << "Radix sort uint key ascending failed at size " << num_items;
        }
    };

    "radix sort pair(uint-float)"_test = [&]
    {
        constexpr int32_t array_size = 524288;
        using radix_key_type         = uint;
        using radix_value_type       = float;
        luisa::vector<radix_key_type>         input_key(array_size);
        luisa::vector<radix_key_type>         input_dec_key(array_size);
        luisa::vector<radix_value_type>       input_value(array_size);
        luisa::vector<radix_value_type>       input_dec_value(array_size);
        std::mt19937                          rng(114521);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for(int i = 0; i < array_size; i++)
        {
            input_key[i]     = static_cast<radix_key_type>(i);
            input_dec_key[i] = static_cast<radix_key_type>(i);
        }
        std::shuffle(input_key.begin(), input_key.end(), rng);
        std::shuffle(input_dec_key.begin(), input_dec_key.end(), rng);
        for(int i = 0; i < array_size; i++)
        {
            input_value[i]     = dist(rng);
            input_dec_value[i] = dist(rng);
        }
        // pair
        luisa::vector<std::pair<radix_key_type, radix_value_type>> key_value_pairs(array_size);
        luisa::vector<std::pair<radix_key_type, radix_value_type>> dec_key_value_pairs(array_size);
        for(int i = 0; i < array_size; i++)
        {
            key_value_pairs[i]     = std::make_pair(input_key[i], input_value[i]);
            dec_key_value_pairs[i] = std::make_pair(input_dec_key[i], input_dec_value[i]);
        }


        auto key_buffer = device.create_buffer<radix_key_type>(array_size);
        stream << key_buffer.copy_from(input_key.data()) << synchronize();

        auto value_buffer = device.create_buffer<radix_value_type>(array_size);
        stream << value_buffer.copy_from(input_value.data()) << synchronize();

        auto key_out_buffer   = device.create_buffer<radix_key_type>(array_size);
        auto value_out_buffer = device.create_buffer<radix_value_type>(array_size);
        radixsorter.SortPairs<radix_key_type, radix_value_type>(cmdlist,
                                                                stream,
                                                                key_buffer.view(),
                                                                key_out_buffer.view(),
                                                                value_buffer.view(),
                                                                value_out_buffer.view(),
                                                                key_buffer.size());

        // sec
        luisa::vector<radix_key_type> result(array_size);
        stream << key_out_buffer.copy_to(result.data()) << synchronize();
        luisa::vector<radix_value_type> value_result(array_size);
        stream << value_out_buffer.copy_to(value_result.data()) << synchronize();

        auto dec_key_in_buffer = device.create_buffer<radix_key_type>(array_size);
        stream << dec_key_in_buffer.copy_from(input_dec_key.data()) << synchronize();
        auto dec_key_out_buffer = device.create_buffer<radix_key_type>(array_size);

        // desc
        auto dec_value_buffer = device.create_buffer<radix_value_type>(array_size);
        stream << dec_value_buffer.copy_from(input_dec_value.data()) << synchronize();
        auto dec_value_out_buffer = device.create_buffer<radix_value_type>(array_size);
        radixsorter.SortPairsDescending<radix_key_type, radix_value_type>(cmdlist,
                                                                          stream,
                                                                          dec_key_in_buffer.view(),
                                                                          dec_key_out_buffer.view(),
                                                                          dec_value_buffer.view(),
                                                                          dec_value_out_buffer.view(),
                                                                          dec_key_in_buffer.size());

        luisa::vector<radix_key_type> desc_result(array_size);
        stream << dec_key_out_buffer.copy_to(desc_result.data()) << synchronize();
        luisa::vector<radix_value_type> desc_value_result(array_size);
        stream << dec_value_out_buffer.copy_to(desc_value_result.data()) << synchronize();

        // expect
        std::sort(key_value_pairs.begin(),
                  key_value_pairs.end(),
                  [](const std::pair<radix_key_type, radix_value_type>& a,
                     const std::pair<radix_key_type, radix_value_type>& b)
                  { return a.first < b.first; });
        std::sort(dec_key_value_pairs.begin(),
                  dec_key_value_pairs.end(),
                  [](const std::pair<radix_key_type, radix_value_type>& a,
                     const std::pair<radix_key_type, radix_value_type>& b)
                  { return a.first > b.first; });

        bool pass =
            std::equal(result.begin(),
                       result.end(),
                       key_value_pairs.begin(),
                       [](const radix_key_type& a, const std::pair<radix_key_type, radix_value_type>& b)
                       { return a == b.first; });
        bool value_pass =
            std::equal(value_result.begin(),
                       value_result.end(),
                       key_value_pairs.begin(),
                       [](const radix_value_type& a, const std::pair<radix_key_type, radix_value_type>& b)
                       { return a == b.second; });

        expect(pass && value_pass) << "Radix sort uint-float pair ascending key failed at size " << array_size;

        bool desc_pass =
            std::equal(desc_result.begin(),
                       desc_result.end(),
                       dec_key_value_pairs.begin(),
                       [](const radix_key_type& a, const std::pair<radix_key_type, radix_value_type>& b)
                       { return a == b.first; });
        bool desc_value_pass =
            std::equal(desc_value_result.begin(),
                       desc_value_result.end(),
                       dec_key_value_pairs.begin(),
                       [](const radix_value_type& a, const std::pair<radix_key_type, radix_value_type>& b)
                       { return a == b.second; });
        expect(desc_pass && desc_value_pass)
            << "Radix sort uint-float pair descending key failed at size " << array_size;
    };

    return 0;
}
