add_executable(sirius_tests)

target_sources(sirius_tests
    PRIVATE
        test_tiff_io.cpp
        test_tiff_file.cpp
        test_buffer.cpp
        test_tensor_util.cpp
        test_fft_util.cpp
        test_fft1d.cpp
        test_fft2d.cpp
        test_fft3d.cpp
        test_real_fft.cpp
        test_fft_cuda.cpp
        test_real_fft_cuda.cpp
        test_sim_parameters.cpp
        test_separation.cpp
        test_reconstruction.cpp
        test_otf.cpp
        test_preprocess.cpp
)

target_link_libraries(sirius_tests
    PRIVATE
        sirius
        Catch2::Catch2WithMain
        TIFF::TIFF
)

target_compile_definitions(sirius_tests
    PRIVATE
        SIRIUS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

# Windows has no RPATH: the nvTIFF/nvCOMP runtime DLLs live in their
# FetchContent trees, so put copies next to the test binary or it cannot
# even start (0xC0000135) -- which also breaks Catch2 test discovery.
if(WIN32 AND SIRIUS_NVTIFF_RUNTIME_LIBS)
    foreach(_runtime_dll IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        add_custom_command(TARGET sirius_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_runtime_dll}" "$<TARGET_FILE_DIR:sirius_tests>"
            VERBATIM)
    endforeach()
endif()

include(Catch)
# GPU-only cases SKIP at run time when no CUDA device is usable, so the same
# binary passes on CPU-only CI. `ctest -L cuda` / the [cuda] tag select them.
catch_discover_tests(sirius_tests TEST_PREFIX "sirius::")

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius_tests)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_tests)
endif()
