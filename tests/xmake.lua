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
                if not os.exists(target:targetdir()) then
                    os.mkdir(target:targetdir())
                end
                if is_host("windows") then
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dll"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc.exe"), target:targetdir())
                elseif is_host("macosx") then
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dylib"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc"), target:targetdir())
                else
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.so"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc"), target:targetdir())
                end
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
  add_test_target("host_enqueue_probe")
  -- Epoch D: compile-time policy sweep harness (measurement/evidence tool; not
  -- part of the regression pass criteria).
  add_test_target("policy_tune")
add_test_target("vk_cuda_interop_reduce_test")
add_test_target("vk_cuda_interop_ops_test")

-- CUDA environment for the CUB baseline benchmark (nvcc + CUDA headers/libs).
-- Mirrors LuisaCompute's cuda_ext rule; disables the targets when CUDA is absent.
rule("lcpp_cuda_env")
on_load(function(target)
    local cuda_path = os.getenv("CUDA_PATH")
    if cuda_path then
        target:add("sysincludedirs", path.join(cuda_path, "include"), {public = true})
        target:add("linkdirs", path.join(cuda_path, "lib/x64/"), {public = true})
        target:add("links", "cudart", "cuda", {public = true})
    else
        target:set("enabled", false)
    end
end)
rule_end()

-- CUB baseline kernels, compiled by nvcc (extern "C" wrappers around NVIDIA CUB).
target("lcpp_cub_bench_kernels")
    set_kind("static")
    add_rules("lcpp_cuda_env")
    set_toolchains("cuda")
    set_policy("build.cuda.devlink", true)
    set_languages("cxx17")
    set_runtimes("MD")
    add_files("cub_bench_kernels.cu")
    add_cugencodes("native")
    -- CCCL headers require the conforming MSVC preprocessor on the host side
    add_cuflags("-Xcompiler=/Zc:preprocessor", {force = true})
target_end()

-- Benchmark: lcpp device primitives vs NVIDIA CUB on the same GPU.
-- usage: lcpp_cub_bench [cuda|dx|vk|...] [num_items] [repetitions]
target("lcpp_cub_bench")
    set_kind("binary")
    add_files("lcpp_cub_bench.cpp")
    set_languages("c++20")
    add_deps("lcpp")
    add_rules("lcpp_cuda_env")
    add_deps("lcpp_cub_bench_kernels")
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
target_end()