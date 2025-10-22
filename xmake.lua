set_xmakever("3.0.0")
set_version("0.0.1")

add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

option("test", {default = true, description = "build test example", type = "boolean"})
includes("xmake/package.lua")

add_requires("luisa-compute", {configs = {cuda = true}})

target("lcpp")
    set_kind("headeronly")
    add_headerfiles("src/(lcpp/**.h)", {public = true})
    add_includedirs("src/", {public = true})
    add_packages("luisa-compute", {public = true})
target_end()

includes("tests")