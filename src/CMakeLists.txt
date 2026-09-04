add_library(sirius STATIC)

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(sirius
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_sources(sirius
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ${PROJECT_SOURCE_DIR}/include
        FILES
            ${PROJECT_SOURCE_DIR}/include/sirius/device.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/buffer.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/tiff_io.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/tensor_util.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft_common.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/real_fft.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft_util.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/sim_parameters.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/legacy_config.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/otf.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/preprocess.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/separation.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/illumination.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/sim_reconstruction.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/registration.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/stitching.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/constants.hpp
    PRIVATE
        device.cpp
        buffer.cpp
        tiff_io.cpp
        fft.cpp
        real_fft.cpp
        sim_parameters.cpp
        legacy_config.cpp
        otf.cpp
        preprocess.cpp
        separation.cpp
        sim_cpu.cpp
        sim_reconstruction.cpp
        registration.cpp
        stitching.cpp
)

target_link_libraries(sirius
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        TIFF::TIFF
        OpenMP::OpenMP_CXX
        ${SIRIUS_FFTW_TARGETS}
        sirius_simd
        tomlplusplus::tomlplusplus
)

# The public headers never depend on these macros (see device.hpp); they only
# select implementation paths inside the library.
if(SIRIUS_ENABLE_CUDA)
    target_compile_definitions(sirius PRIVATE SIRIUS_HAS_CUDA=1)
    target_sources(sirius PRIVATE
        cuda/kernels.cu
        cuda/sim_kernels.cu
        fft_cufft.cpp
    )
    target_link_libraries(sirius PRIVATE CUDA::cudart_static CUDA::cufft)
    if(SIRIUS_ENABLE_NVTIFF)
        target_compile_definitions(sirius PRIVATE SIRIUS_HAS_NVTIFF=1)
        target_sources(sirius PRIVATE tiff_nvtiff.cpp)
        target_link_libraries(sirius PRIVATE nvtiff::nvtiff)
        if(SIRIUS_ENABLE_NVCOMP)
            target_link_libraries(sirius PRIVATE sirius_nvcomp_runtime)
        endif()
    endif()
endif()

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius)
endif()
