set(SIRIUS_BINDING_SOURCES
    src/module.cpp
    src/bind_device.cpp
    src/bind_buffer.cpp
    src/bind_fft.cpp
    src/bind_tiff_io.cpp
    src/bind_sim.cpp
    src/bind_registration.cpp
)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  _sirius_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ${SIRIUS_BINDING_SOURCES}
)

target_link_libraries(_sirius_ext PRIVATE sirius sirius_simd)
target_include_directories(_sirius_ext PRIVATE ${PROJECT_SOURCE_DIR}/include)

# The pure-Python sources are shipped via `wheel.packages` in pyproject.toml,
# which is also what makes `pip install -e .` editable for Python files.
# CMake only owns the compiled extension and any runtime DLLs.
install(TARGETS _sirius_ext LIBRARY DESTINATION sirius COMPONENT python)
install(FILES $<TARGET_RUNTIME_DLLS:_sirius_ext> DESTINATION sirius COMPONENT python)

# nvTIFF / nvCOMP are prebuilt shared libraries: ship them next to the
# extension and let it find them via $ORIGIN, so an installed wheel does not
# depend on the (gitignored) build tree. cuFFT/cudart are resolved from the
# CUDA toolkit on the machine, as with any CUDA-enabled Python package.
if(SIRIUS_ENABLE_NVTIFF AND NOT WIN32)
    set_target_properties(_sirius_ext PROPERTIES
        INSTALL_RPATH "$ORIGIN;${CUDAToolkit_LIBRARY_DIR}"
        BUILD_WITH_INSTALL_RPATH OFF)
    # The redistributables ship "libnvtiff.so.0 -> libnvtiff.so.0.8.0.82"
    # symlinks; install(FILES) would copy the dangling link, so install the
    # real file under the soname the extension's DT_NEEDED entry refers to.
    foreach(_lib IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        get_filename_component(_soname "${_lib}" NAME)
        file(REAL_PATH "${_lib}" _real)
        install(FILES "${_real}" DESTINATION sirius COMPONENT python RENAME "${_soname}")
    endforeach()
endif()
