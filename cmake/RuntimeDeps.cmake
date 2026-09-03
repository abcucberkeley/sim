# Windows has no RPATH: the prebuilt nvTIFF/nvCOMP runtime DLLs live in their
# FetchContent trees, so an executable that links sirius needs copies next to
# it or it cannot even start (0xC0000135) -- which also breaks Catch2 test
# discovery. No-op elsewhere and for CPU-only builds.
function(sirius_copy_runtime_dlls tgt)
    if(NOT WIN32 OR NOT SIRIUS_NVTIFF_RUNTIME_LIBS)
        return()
    endif()
    foreach(_runtime_dll IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_runtime_dll}" "$<TARGET_FILE_DIR:${tgt}>"
            VERBATIM)
    endforeach()
endfunction()
