
add_requires("boost_ut","cpptrace")
local function add_test_target(file_name)
    target(file_name)
        set_kind("binary")
        add_files(file_name..".cpp")
        set_languages("c++20")
        add_deps("lcpp")
        add_packages("boost_ut","cpptrace")
        on_load(function(target)
            if has_config('lc_use_xrepo') then
                target:add('packages', "luisa-compute", {public = true})
            else
                target:add('deps', 'lc-runtime', 'lc-dsl')
            end

            if target:is_plat('macosx') then
                target:add('defines', '__APPLE__')
            end
        end)
        -- add run path for luisa-compute
        on_config(function (target)
            if has_config('lc_use_xrepo') then
                target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
            end
        end)
    target_end()
end

add_test_target("block_level_test")
add_test_target("warp_level_test")
add_test_target("decoupled_look_back")
add_test_target("device_reduce_test")
add_test_target("device_scan_test")
add_test_target("device_segment_reduce")
add_test_target("device_radix_sort_one_sweep")