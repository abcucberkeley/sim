include(FetchContent)

# Static deps must be PIC-compatible when linked into the Python extension (.so)
if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Eigen3 (header-only). Its CMake project is deliberately NOT added:
# Eigen 3.4's CMakeLists probes OpenGL, Python and -- through the legacy
# FindCUDA module in unsupported/test -- the system CUDA libraries, which
# pre-seeds CUDA_cufft_LIBRARY & co. in the cache with whatever distro toolkit
# lives in /usr/lib before our FindCUDAToolkit runs. SOURCE_SUBDIR points at a
# directory without a CMakeLists.txt, so FetchContent only downloads the
# sources and we describe the target ourselves.
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake-not-used
)
FetchContent_MakeAvailable(Eigen3)
add_library(sirius_eigen INTERFACE)
# SYSTEM: excluded from warnings and MSVC /analyze, like the other deps.
target_include_directories(sirius_eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}>)
add_library(Eigen3::Eigen ALIAS sirius_eigen)

# zlib — provides the DEFLATE/ZIP codec for libtiff. Without it, libtiff's
# internal find_package(ZLIB) fails and ZIP_SUPPORT is left undefined, so
# writing a TIFF with TiffCompression::Deflate fails at encode time.
# OVERRIDE_FIND_PACKAGE redirects that find_package(ZLIB) to this fetched copy.
FetchContent_Declare(
    ZLIB
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        v1.3.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # zlib targets an old CMake floor
    set(ZLIB_BUILD_EXAMPLES OFF)
    FetchContent_MakeAvailable(ZLIB)
    # zlib's CMake exports zlibstatic/zlib but not the canonical ZLIB::ZLIB
    # target libtiff links against. Create it from the static lib (PIC is on
    # for the Python extension) and ensure its headers are on the usage
    # interface: zlib.h lives in the source tree, generated zconf.h in the
    # build tree.
    if(NOT TARGET ZLIB::ZLIB)
        target_include_directories(zlibstatic PUBLIC
            $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>
            $<BUILD_INTERFACE:${zlib_BINARY_DIR}>)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
endblock()

# libtiff
FetchContent_Declare(
    libtiff
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG        v4.7.0
    GIT_SHALLOW    TRUE
)
# compatibility with cmake < 3.5 has been removed from CMake
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(tiff-tools   OFF)
    set(tiff-tests   OFF)
    set(tiff-contrib OFF)
    set(tiff-docs    OFF)
    FetchContent_MakeAvailable(libtiff)
    if(NOT TARGET TIFF::TIFF)
        add_library(TIFF::TIFF ALIAS tiff)
    endif()
endblock()

# FFTW3
FetchContent_Declare(
    fftw3
    URL https://www.fftw.org/fftw-3.3.10.tar.gz
)
# fftw using offensive global names
block()
    # FFTW 3.3.10 declares cmake_minimum_required(VERSION 3.0); CMake 4.x
    # removed support for <3.5, so spoof a 3.5 floor for this subtree only.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    # FFTW3 uses cmake_minimum_required(3.0), so CMP0077 defaults OLD and option()
    # ignores normal variables; NEW makes it honor our BUILD_SHARED_LIBS=OFF below.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTS OFF) # Build tests
    set(ENABLE_OPENMP  ON) # Use OpenMP for multithreading
    set(ENABLE_THREADS OFF) # Use pthread for multithreading
    set(ENABLE_FLOAT OFF) # single-precision (unused; sirius uses double fftw_* API)
    set(ENABLE_LONG_DOUBLE OFF) # long-double precision
    set(ENABLE_QUAD_PRECISION OFF) # quadruple-precision
    set(ENABLE_SSE OFF)
    set(ENABLE_SSE2  ${SIRIUS_ENABLE_SSE2})
    set(ENABLE_AVX   ${SIRIUS_ENABLE_AVX})
    set(ENABLE_AVX2  ${SIRIUS_ENABLE_AVX2})
    set(ENABLE_AVX512 ${SIRIUS_ENABLE_AVX512})
    FetchContent_MakeAvailable(fftw3)
    target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
    add_library(FFTW3::fftw3 ALIAS fftw3)
    # if omp is enabled, create fftw3_omp alias for the target
    if(TARGET fftw3_omp)
        add_library(FFTW3::fftw3_omp ALIAS fftw3_omp)
    endif()
endblock()

# Canonical FFTW link targets: the core double-precision lib, plus the OpenMP
# threading lib when FFTW was built with ENABLE_OPENMP. Consumers link
# ${SIRIUS_FFTW_TARGETS} rather than repeating this conditional.
set(SIRIUS_FFTW_TARGETS FFTW3::fftw3)
if(TARGET FFTW3::fftw3_omp)
    list(APPEND SIRIUS_FFTW_TARGETS FFTW3::fftw3_omp)
endif()

# OpenMP (provided by the host compiler)
find_package(OpenMP REQUIRED)

# toml++ : TOML parser/serializer
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM          # mark its headers as system -> excluded from warnings/analyze
)
FetchContent_MakeAvailable(tomlplusplus)

# nlohmann/json: pipeline files, the assistant tool API and the worker
# protocol (header-only; TensorStore uses the same library internally).
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    SYSTEM
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# TensorStore (zarr / N5). Its CMake bridge normally fetches private copies of
# zlib, libtiff and nlohmann/json under the same target names we already
# define (ZLIB::ZLIB, TIFF::TIFF, nlohmann_json::nlohmann_json), and two zlibs
# in one static link would clash anyway. So those three are declared "system"
# packages for TensorStore and find_package() is redirected to the targets
# built above; everything else (abseil, blosc, zstd, riegeli, ...) is fetched
# and built by the bridge, out of sight.
if(SIRIUS_ENABLE_TENSORSTORE)
    # The bridge enables the ASM_NASM language (libjpeg-turbo / BoringSSL).
    # Look where package managers put nasm when it is not on PATH; conda-forge's
    # `conda install nasm` is the no-sudo route on a shared machine.
    if(NOT DEFINED CMAKE_ASM_NASM_COMPILER AND NOT DEFINED ENV{ASM_NASM})
        find_program(SIRIUS_NASM NAMES nasm
            HINTS ENV CONDA_PREFIX "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3" "$ENV{HOME}/mambaforge"
                  "$ENV{HOME}/miniforge3" /opt/conda /usr/local
            PATH_SUFFIXES bin)
        if(SIRIUS_NASM)
            set(CMAKE_ASM_NASM_COMPILER "${SIRIUS_NASM}" CACHE FILEPATH "NASM assembler for TensorStore's dependencies" FORCE)
        else()
            message(FATAL_ERROR "SIRIUS_ENABLE_TENSORSTORE needs the NASM assembler (apt install nasm, "
                                "conda install -c conda-forge nasm, or set CMAKE_ASM_NASM_COMPILER).")
        endif()
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)   # bazel_to_cmake

    set(TENSORSTORE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_ZLIB ON CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_TIFF ON CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_NLOHMANN_JSON ON CACHE BOOL "" FORCE)
    # find_package(TIFF) / find_package(nlohmann_json) inside the bridge must
    # resolve to our targets: drop config files into the redirects directory
    # CMake consults before any module or installed package (zlib already has
    # one from OVERRIDE_FIND_PACKAGE above).
    foreach(_pkg TIFF nlohmann_json)
        string(TOLOWER "${_pkg}" _lc)
        file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${_lc}-config.cmake"
             "# ${_pkg} is built in-tree by SIRIUS (cmake/Dependencies.cmake); the targets already exist.\n"
             "set(${_pkg}_FOUND TRUE)\n")
        file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${_pkg}Config.cmake"
             "include(\"\${CMAKE_CURRENT_LIST_DIR}/${_lc}-config.cmake\")\n")
    endforeach()
    set(TIFF_LIBRARIES TIFF::TIFF)
    set(TIFF_INCLUDE_DIRS "")

    FetchContent_Declare(
        tensorstore
        URL      https://github.com/google/tensorstore/archive/refs/tags/v${SIRIUS_TENSORSTORE_VERSION}.tar.gz
        URL_HASH SHA256=f59667a32357b8cc0c752429927ad97654f6a67c7d3d62b9efea902c6798d473
        SYSTEM
    )
    FetchContent_MakeAvailable(tensorstore)
    # The drivers the library uses. all_drivers would also pull gcs/s3/http.
    add_library(sirius_tensorstore INTERFACE)
    target_link_libraries(sirius_tensorstore INTERFACE
        tensorstore::tensorstore
        tensorstore::cast
        tensorstore::index_space_dim_expression
        tensorstore::driver_cast
        tensorstore::driver_zarr
        tensorstore::driver_zarr3
        tensorstore::driver_n5
        tensorstore::kvstore_file)
    add_library(sirius::tensorstore ALIAS sirius_tensorstore)
    message(STATUS "TensorStore ${SIRIUS_TENSORSTORE_VERSION} (zarr v2/v3, N5) enabled")
endif()

if(SIRIUS_ENABLE_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    find_package(Python 3.9
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        a835245fa0c8f6c8d06a25713562100464e95039
        # Fix an upstream MSVC build error in the Eigen::Tensor caster
        # (std::array<long, N> vs Eigen::Index). See the patch script for details.
        PATCH_COMMAND  ${CMAKE_COMMAND} -P
                       ${CMAKE_CURRENT_LIST_DIR}/patches/fix_nanobind_tensor.cmake
    )
    FetchContent_MakeAvailable(nanobind)
endif()

if(SIRIUS_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# Qt (GUI application only). Like the CUDA toolkit, Qt is a large prebuilt
# system dependency rather than something FetchContent should build: it is
# found via its CMake package config. Qt 6 is preferred, Qt 5.15 works too.
# Point CMake at an installation with one of
#   -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x/gcc_64        (the *-app-* presets set
#                                                       this from $SIRIUS_QT_DIR)
#   -DQt6_DIR=/path/to/Qt/6.x/gcc_64/lib/cmake/Qt6    (or Qt5_DIR)
# Prefer the Qt*_DIR form for a Qt that lives inside a larger prefix (e.g. a
# conda environment): a prefix path would also expose that environment's
# libjpeg/zstd/... to libtiff's own find_package calls and drag in DLLs.
# The app links the sirius::qt interface target, which resolves to the right
# major version's Widgets module.
if(SIRIUS_ENABLE_APP)
    find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets HINTS ${Qt6_DIR} ${Qt5_DIR})
    if(QT_VERSION_MAJOR EQUAL 6)
        find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets Network)
        set(_sirius_qt_libs Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets Qt6::Network)
    else()
        # Qt 5: QOpenGLWidget lives in Widgets
        find_package(Qt5 REQUIRED COMPONENTS Widgets Network)
        set(_sirius_qt_libs Qt5::Widgets Qt5::Network)
    endif()
    add_library(sirius_qt INTERFACE)
    target_link_libraries(sirius_qt INTERFACE ${_sirius_qt_libs})
    add_library(sirius::qt ALIAS sirius_qt)
    message(STATUS "Qt ${Qt${QT_VERSION_MAJOR}_VERSION} (Widgets, OpenGL, Network) for the SIRIUS app")
endif()

if(SIRIUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "SIRIUS_ENABLE_CUDA=ON but no CUDA compiler was found. "
                            "Set CUDACXX or CMAKE_CUDA_COMPILER to nvcc.")
    endif()
    enable_language(CUDA)
    # Pin the toolkit to the one nvcc came from. Without this FindCUDAToolkit
    # can pick libraries of a second, distro-packaged toolkit that sits in the
    # default linker paths (/usr/lib/x86_64-linux-gnu on Ubuntu), and the
    # binary ends up with cuFFT/cudart from a different CUDA major than the
    # compiler that built the kernels.
    if(NOT DEFINED CUDAToolkit_ROOT)
        get_filename_component(_sirius_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(CUDAToolkit_ROOT "${_sirius_cuda_bin}/.." ABSOLUTE)
    endif()
    find_package(CUDAToolkit 12.0 REQUIRED)
    message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION} (nvcc ${CMAKE_CUDA_COMPILER}), "
                   "architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    # Every CUDA library we link must come from that same toolkit.
    foreach(_lib cufft cudart_static)
        file(REAL_PATH "${CUDA_${_lib}_LIBRARY}" _sirius_lib_real)
        file(REAL_PATH "${CUDAToolkit_LIBRARY_ROOT}" _sirius_root_real)
        string(FIND "${_sirius_lib_real}" "${_sirius_root_real}/" _sirius_pos)
        if(NOT _sirius_pos EQUAL 0)
            message(FATAL_ERROR "CUDA::${_lib} resolved to ${CUDA_${_lib}_LIBRARY}, outside the toolkit "
                                "${CUDAToolkit_LIBRARY_ROOT} that provides nvcc. Clear CUDA_*_LIBRARY cache "
                                "entries (cmake -U 'CUDA_*_LIBRARY') or set CUDAToolkit_ROOT.")
        endif()
    endforeach()
    if(SIRIUS_ENABLE_NVTIFF)
        include(NvidiaRedist)
    endif()
endif()