# NVIDIA redistributable libraries that ship as prebuilt archives rather than
# source: nvTIFF (GPU TIFF codec) and nvCOMP (GPU Deflate, dlopen'ed by nvTIFF).
#
# Both come from https://developer.download.nvidia.com/compute/<lib>/redist/,
# pinned by version + SHA256 (taken from the redistrib_<version>.json manifest)
# so a build is reproducible. SIRIUS_NVTIFF_ROOT / SIRIUS_NVCOMP_ROOT bypass the
# download for machines where the libraries are already installed (HPC modules,
# containers built from containers/install-deps.sh).
#
# Targets provided:
#   nvtiff::nvtiff   shared, INTERFACE_INCLUDE_DIRECTORIES set
#   nvcomp::nvcomp   shared (from nvCOMP's own CMake config)
# Variables provided (for install()/bundling):
#   SIRIUS_NVTIFF_RUNTIME_LIBS, SIRIUS_NVCOMP_RUNTIME_LIBS

include(FetchContent)

set(_NVTIFF_VERSION 0.8.0.82)
set(_NVCOMP_VERSION 5.3.0.16)
set(_NV_REDIST_BASE "https://developer.download.nvidia.com/compute")

# ---- platform / CUDA major -> archive name + hash ---------------------------
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    set(_nv_cuda cuda13)
else()
    set(_nv_cuda cuda12)
endif()

if(WIN32)
    set(_nv_plat windows-x86_64)
    set(_nv_ext zip)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_nv_plat linux-sbsa)
    set(_nv_ext tar.xz)
else()
    set(_nv_plat linux-x86_64)
    set(_nv_ext tar.xz)
endif()

# sha256 from redistrib_0.8.0.json / redistrib_5.3.0.json
set(_nvtiff_sha_linux-x86_64_cuda12  3f1093d321814d6f6264e1dc0a5fc4a799487a68c482ee29d61a924004d06f25)
set(_nvtiff_sha_linux-x86_64_cuda13  619422df0b6fe20282eda842cbba066cc2deb2ed95a87624325f0003984d69bc)
set(_nvtiff_sha_linux-sbsa_cuda12    dea9cb0c4c4c5d20feed219854b2a62e4e73b16e9a95e554613b2c80c07e95ad)
set(_nvtiff_sha_linux-sbsa_cuda13    30573605058e652495ee0a3079570d7ed13ed5530f294f49ea21ac572143a7b2)
set(_nvtiff_sha_windows-x86_64_cuda12 8290e27bdcf2fbf8239269d46eabaa0ecddde51dcd86a9cb2648a3b5824ea551)
set(_nvtiff_sha_windows-x86_64_cuda13 7741f81fccbfc1702ae9eff00684222a778168b9149d5e24897d4710e3776a0c)

set(_nvcomp_sha_linux-x86_64_cuda12  1def6bb0fa51d8ea3fe0c43ae1c58df2f63808ced7444267e14245583ec23f6f)
set(_nvcomp_sha_linux-x86_64_cuda13  2c36f5af63c37e4afe13d14f912e84130e6a05f07b066547b3e028c4ca54c866)
set(_nvcomp_sha_linux-sbsa_cuda12    1c982830d3c0c4b9e6fc55822cfe4f26b56347ddfa29725ee14e1605291bf772)
set(_nvcomp_sha_linux-sbsa_cuda13    d0b42d81db2eed6725058359e6eb287b5d4748dcccc95c93f337bce15fc8c3bf)
set(_nvcomp_sha_windows-x86_64_cuda12 8549b0815770dcbf4d8496dc988fc118f9b9eb58e99a1f74b5b8f31b0e452151)
set(_nvcomp_sha_windows-x86_64_cuda13 f5d8573d6dd48f40c394d19a26bb96fc3b94cd3a6f74fff43dd32e6df141c70d)

# Fetch <lib> unless <root_var> points at an existing install. Sets <out_dir>.
function(_sirius_nv_redist name version root_var sha_prefix out_dir)
    if(${root_var})
        if(NOT EXISTS "${${root_var}}/include")
            message(FATAL_ERROR "${root_var}=${${root_var}} has no include/ directory")
        endif()
        set(${out_dir} "${${root_var}}" PARENT_SCOPE)
        return()
    endif()
    set(_sha "${${sha_prefix}_${_nv_plat}_${_nv_cuda}}")
    if(NOT _sha)
        message(FATAL_ERROR "No ${name} ${version} redistributable is known for ${_nv_plat}/${_nv_cuda}; "
                            "set ${root_var} to an existing install.")
    endif()
    set(_archive "lib${name}-${_nv_plat}-${version}_${_nv_cuda}-archive.${_nv_ext}")
    if(name STREQUAL "nvcomp")
        set(_archive "nvcomp-${_nv_plat}-${version}_${_nv_cuda}-archive.${_nv_ext}")
        set(_url "${_NV_REDIST_BASE}/nvcomp/redist/nvcomp/${_nv_plat}/${_archive}")
    else()
        set(_url "${_NV_REDIST_BASE}/${name}/redist/lib${name}/${_nv_plat}/${_archive}")
    endif()
    FetchContent_Declare(${name}_redist
        URL            "${_url}"
        URL_HASH       SHA256=${_sha}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(${name}_redist)   # no CMakeLists.txt inside: populate only
    set(${out_dir} "${${name}_redist_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

# ---- nvTIFF ------------------------------------------------------------------
_sirius_nv_redist(nvtiff ${_NVTIFF_VERSION} SIRIUS_NVTIFF_ROOT _nvtiff_sha _nvtiff_dir)

add_library(nvtiff::nvtiff SHARED IMPORTED GLOBAL)
set_target_properties(nvtiff::nvtiff PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_nvtiff_dir}/include")
if(WIN32)
    file(GLOB _nvtiff_dll "${_nvtiff_dir}/bin/nvtiff*.dll")
    set_target_properties(nvtiff::nvtiff PROPERTIES
        IMPORTED_LOCATION "${_nvtiff_dll}"
        IMPORTED_IMPLIB   "${_nvtiff_dir}/lib/x64/nvtiff.lib")
    set(SIRIUS_NVTIFF_RUNTIME_LIBS "${_nvtiff_dll}")
else()
    set_target_properties(nvtiff::nvtiff PROPERTIES
        IMPORTED_LOCATION "${_nvtiff_dir}/lib/libnvtiff.so.0"
        IMPORTED_SONAME   "libnvtiff.so.0")
    set(SIRIUS_NVTIFF_RUNTIME_LIBS "${_nvtiff_dir}/lib/libnvtiff.so.0")
endif()
# nvTIFF is a plain C library that dlopen()s libcuda.so.1 itself; headers need
# cuda_runtime.h + library_types.h from the toolkit.
target_link_libraries(nvtiff::nvtiff INTERFACE CUDA::cudart_static)
message(STATUS "nvTIFF ${_NVTIFF_VERSION} (${_nv_plat}/${_nv_cuda}): ${_nvtiff_dir}")

# ---- nvCOMP (optional; only needed for Deflate/ZIP compressed TIFFs) ---------
set(SIRIUS_NVCOMP_RUNTIME_LIBS "")
if(SIRIUS_ENABLE_NVCOMP)
    _sirius_nv_redist(nvcomp ${_NVCOMP_VERSION} SIRIUS_NVCOMP_ROOT _nvcomp_sha _nvcomp_dir)
    find_package(nvcomp CONFIG REQUIRED PATHS "${_nvcomp_dir}" NO_DEFAULT_PATH)
    if(WIN32)
        file(GLOB SIRIUS_NVCOMP_RUNTIME_LIBS "${_nvcomp_dir}/bin/nvcomp*.dll")
    else()
        set(SIRIUS_NVCOMP_RUNTIME_LIBS "${_nvcomp_dir}/lib/libnvcomp.so.5")
    endif()
    message(STATUS "nvCOMP ${_NVCOMP_VERSION}: ${_nvcomp_dir}")

    # nvTIFF loads libnvcomp.so.5 with dlopen() rather than linking it, so
    # nothing in our link line references a symbol from it. GNU ld's default
    # --as-needed would then drop the DT_NEEDED entry and nvTIFF would report
    # NVTIFF_STATUS_NVCOMP_NOT_FOUND at runtime. This link-library feature keeps
    # the entry (and the RPATH CMake derives from it) so the dynamic loader has
    # libnvcomp resident before nvTIFF asks for it.
    set(CMAKE_LINK_LIBRARY_USING_SIRIUS_KEEP_NEEDED
        "LINKER:--push-state,--no-as-needed" "<LINK_ITEM>" "LINKER:--pop-state")
    set(CMAKE_LINK_LIBRARY_USING_SIRIUS_KEEP_NEEDED_SUPPORTED TRUE)
    add_library(sirius_nvcomp_runtime INTERFACE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_link_libraries(sirius_nvcomp_runtime INTERFACE
            "$<LINK_LIBRARY:SIRIUS_KEEP_NEEDED,nvcomp::nvcomp>")
    else()
        target_link_libraries(sirius_nvcomp_runtime INTERFACE nvcomp::nvcomp)
    endif()
endif()
