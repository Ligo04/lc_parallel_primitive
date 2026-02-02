set_xmakever("3.0.0")
set_version("0.0.1")

add_rules("mode.release", "mode.debug", "mode.releasedbg")

option("lcpp_test", {default = true, description = "build test example", type = "boolean"})
option('lc_use_xrepo', {default = false})

if has_config('lc_use_xrepo') then
    includes("xmake/package.lua")
    add_requires("luisa-compute", {configs = {cuda = true}})
else
    -- Change LCPP_LC_DIR for submod path
    LCPP_LC_DIR = 'LuisaCompute'
    local lc_lcpp_options = {
        lc_enable_xir = true,
        lc_enable_tests = false,
    }
    if lc_options then
        for k,v in pairs(lc_lcpp_options) do
            lc_options[k] = v
        end
    else
        lc_options = lc_lcpp_options
    end
    if LCPP_LC_DIR and os.exists(LCPP_LC_DIR) then
        includes(LCPP_LC_DIR)
    end
end
target("lcpp")
    set_kind("headeronly")
    set_languages("c++20")
    add_headerfiles("src/(lcpp/**.h)", {public = true})
    add_includedirs("src/", {public = true})
    on_load(function(target)
        if has_config('lc_use_xrepo') then
            target:add('packages', "luisa-compute", {public = true})
        else
            target:add('deps', 'lc-runtime', 'lc-dsl')
        end
    end)
target_end()

if has_config('"lcpp_test"') then
    includes("tests")
end