include(FetchContent)

# boost ut
FetchContent_Declare(
        boost_ut
        GIT_REPOSITORY "https://github.com/boost-ext/ut.git"
        GIT_TAG "master"
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(boost_ut)

# cpptrace
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY "https://github.com/jeremy-rifkin/cpptrace"
        GIT_TAG "main"
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cpptrace)

enable_testing()
function(lcpp_add_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE lcpp Boost::ut cpptrace::cpptrace)
    add_test(NAME ${test_name} COMMAND ${test_name} $<TARGET_FILE_DIR:luisa-compute-core>)
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "LUISA_EXPERIMENTAL_LLVM_CODEGEN=1")
endfunction()

lcpp_add_test(block_level_test)
lcpp_add_test(decoupled_look_back)
lcpp_add_test(device_reduce_test)
lcpp_add_test(device_scan_test)
lcpp_add_test(warp_level_test)
