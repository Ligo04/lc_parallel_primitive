
add_requires("boost_ut")
add_requires("cpptrace")
local function add_test_target(file_name)
    target(file_name)
        set_kind("binary")
        add_files(file_name..".cpp")
        add_deps("lcpp")
        add_packages("boost_ut","luisa-compute","cpptrace")
        -- add run path for luisa-compute
        if is_os("mac") then
            add_defines("__APPLE__")
        end
        on_config(function (target)
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        end)
    target_end()
end

add_test_target("device_level_test")
add_test_target("block_level_test")
add_test_target("warp_level_test")
add_test_target("decoupled_look_back")