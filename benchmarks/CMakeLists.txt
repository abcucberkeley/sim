# C++ benchmarks (TIFF reading, SIM reconstruction). Gated by
# SIRIUS_ENABLE_BENCHMARKS at the top level.
#
# Only the sirius TIFF bench is built here. The cpp-tiff bench is built
# standalone by setup_cpptiff.sh, because cpp-tiff bundles its own static
# libtiff under a target also named `tiff` -- adding it to this build would
# collide with the `tiff` target SIRIUS pulls in via FetchContent and risk
# duplicate-libtiff ODR.

foreach(_bench IN ITEMS bench_tiff_sirius bench_sim)
    add_executable(${_bench} ${_bench}.cpp)
    target_include_directories(${_bench} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${_bench} PRIVATE sirius OpenMP::OpenMP_CXX)
    sirius_copy_runtime_dlls(${_bench})
    if(SIRIUS_ENABLE_WARNINGS)
        target_set_warnings(${_bench})
    endif()
    if(SIRIUS_ENABLE_SANITIZERS)
        target_enable_sanitizers(${_bench})
    endif()
endforeach()
