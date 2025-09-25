set_xmakever("3.0.0")
set_version("0.0.1")

add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

option("test", {default = true, description = "build test example", type = "boolean"})
includes("xmake/package.lua")

add_requires("luisa-compute", {configs = {cuda = true}})

target("lc_parallel_primitive")
    set_kind("headeronly")
    add_headerfiles("src/(lc_parallel_primitive/**.h)", {public = true})
    add_includedirs("src/", {public = true})
    add_packages("luisa-compute", {public = true})
target_end()


if has_config("test") then
    add_requires("boost_ut")
    target("test")
        set_kind("binary")
        add_files("tests/**.cpp")
        add_deps("lc_parallel_primitive")
        add_packages("boost_ut","luisa-compute")
        -- add run path for luisa-compute
        if is_os("mac") then
            add_defines("__APPLE__")
        end

        on_config(function (target)
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        end)
    target_end()
end