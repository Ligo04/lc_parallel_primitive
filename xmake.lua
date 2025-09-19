set_xmakever("3.0.0")
set_version("0.0.1")

add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")
set_policy("build.compile_commands", true)

option("test", {default = true, description = "build test example", type = "boolean"})

add_requires("luisa-compute 3ab5163b1736b96b61627eff9f81b9c49f1e2918", {configs = {cuda = true}})

target("lc_parallel_primitive")
    set_kind("headeronly")
    add_headerfiles("src/(lc_parallel_primitive/**.h)", {public = true})
    add_includedirs("src/", {public = true})
    add_packages("luisa-compute", {public = true})

    -- add run path for luisa-compute
    on_config(function (target)
        target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
    end)
target_end()

if has_config("test") then
    target("test")
        set_kind("binary")
        add_files("tests/**.cpp")
        add_deps("lc_parallel_primitive")
        add_packages("luisa-compute")
        -- add run path for luisa-compute
        on_config(function (target)
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        end)
    target_end()
end