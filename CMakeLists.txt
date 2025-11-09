cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(lcpp)

if (NOT TARGET luisa::compute)
    # download LuisaCompute from github.com/LuisaGroup/LuisaCompute.git to the build directory
    include(FetchContent)
    FetchContent_Populate(luisa_compute
            GIT_REPOSITORY "https://github.com/LuisaGroup/LuisaCompute.git"
            GIT_TAG next
            GIT_SHALLOW TRUE)
    set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")

    include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    # add LuisaCompute as a subdirectory
    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_GUI ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_WAYLAND ON CACHE BOOL "" FORCE)
    set(LUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN ON CACHE BOOL "" FORCE)
    add_subdirectory(${LuisaCompute_SOURCE_DIR})
endif ()

add_library(lcpp INTERFACE)
target_include_directories(lcpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lcpp INTERFACE luisa::compute)

option(LCPP_BUILD_TESTS "Build lcpp tests" ${lcpp_IS_TOP_LEVEL})

if (LCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
