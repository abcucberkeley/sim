option(SIRIUS_ENABLE_MPI "Enable MPI" OFF)
option(SIRIUS_ENABLE_CUDA "Enable CUDA (device buffers, cuFFT, nvTIFF)" OFF)
option(SIRIUS_ENABLE_PYTHON_BINDINGS "Enable nanobind python bindings" OFF)

# nvTIFF decodes TIFF strips/tiles straight into device memory. It is an NVIDIA
# redistributable (no source), fetched from developer.download.nvidia.com by
# cmake/Dependencies.cmake, or taken from SIRIUS_NVTIFF_ROOT when set (e.g. a
# cluster module). Deflate/ZIP decoding additionally needs nvCOMP at runtime.
include(CMakeDependentOption)
cmake_dependent_option(SIRIUS_ENABLE_NVTIFF "Enable GPU TIFF decoding via nvTIFF" ON
                       "SIRIUS_ENABLE_CUDA" OFF)
cmake_dependent_option(SIRIUS_ENABLE_NVCOMP "Fetch nvCOMP so nvTIFF can decode Deflate/ZIP TIFFs on the GPU" ON
                       "SIRIUS_ENABLE_NVTIFF" OFF)
set(SIRIUS_NVTIFF_ROOT "" CACHE PATH "Existing nvTIFF install (include/ and lib/) to use instead of downloading")
set(SIRIUS_NVCOMP_ROOT "" CACHE PATH "Existing nvCOMP install (include/ and lib/) to use instead of downloading")

# scikit-build-core always builds the Python extension
if(SKBUILD)
    set(SIRIUS_ENABLE_PYTHON_BINDINGS ON CACHE BOOL "" FORCE)
endif()
option(SIRIUS_ENABLE_SSE2   "Enable SSE2 instruction set"    OFF)
option(SIRIUS_ENABLE_AVX    "Enable AVX instruction set"     OFF)
option(SIRIUS_ENABLE_AVX2   "Enable AVX2 + FMA instruction sets" OFF)
option(SIRIUS_ENABLE_AVX512 "Enable AVX-512F + FMA instruction sets" OFF)

# Development related options
option(SIRIUS_ENABLE_TESTS "Enable tests" OFF)
option(SIRIUS_ENABLE_BENCHMARKS "Build C++ benchmarks" OFF)
option(SIRIUS_ENABLE_WARNINGS "Enable extra warnings" OFF)
option(SIRIUS_ENABLE_SANITIZERS "Enable sanitizers (Debug, non-MSVC)" OFF)
option(SIRIUS_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(SIRIUS_ENABLE_CPPCHECK "Enable cppcheck" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SIRIUS_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Distro-packaged nvcc (/usr/bin/nvcc) is frequently older than the host
    # compiler supports. Prefer the toolkit a module or the CUDA installer put
    # in CUDA_HOME / CUDA_PATH / /usr/local/cuda, unless the user pinned one.
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        find_program(_sirius_nvcc NAMES nvcc
            HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT /usr/local/cuda
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
        if(_sirius_nvcc)
            set(CMAKE_CUDA_COMPILER "${_sirius_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()
    # "native" = the GPUs in this machine (dev builds). Release presets pass an
    # explicit list so the binary runs on the cluster's cards too.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

# Symlink the build-dir compile_commands.json for IDE integration
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json" FORCE)
    # Silently ignore failures (e.g. Windows without Developer Mode enabled)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
        RESULT_VARIABLE _symlink_result
        ERROR_QUIET
        OUTPUT_QUIET
    )
endif()
