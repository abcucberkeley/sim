# Qt GUI for SIM reconstruction. Gated by SIRIUS_ENABLE_APP at the top level
# (Qt is located in cmake/Dependencies.cmake as the sirius::qt target).
#
# Two targets:
#   sirius_app_core  static, Qt-free: session model and display mapping.
#                    Unit-tested by tests/test_app_core.cpp, so the logic the
#                    GUI relies on never needs a display to be verified.
#   sirius-app       the Widgets executable (app/qt/*).

# --- core -------------------------------------------------------------------
add_library(sirius_app_core STATIC
    core/display_mapping.cpp
    core/session.cpp
    core/volume_ops.cpp
)
add_library(sirius::app_core ALIAS sirius_app_core)
target_include_directories(sirius_app_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sirius_app_core PUBLIC sirius)
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius_app_core)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_app_core)
endif()

# --- GUI --------------------------------------------------------------------
set(CMAKE_AUTOMOC ON)

add_executable(sirius-app WIN32 MACOSX_BUNDLE
    qt/main.cpp
    qt/main_window.cpp
    qt/main_window.hpp
    qt/parameter_panel.cpp
    qt/parameter_panel.hpp
    qt/qt_strings.hpp
    qt/image_canvas.cpp
    qt/image_canvas.hpp
    qt/stack_view.cpp
    qt/stack_view.hpp
    qt/band_view.cpp
    qt/band_view.hpp
    qt/recon_worker.cpp
    qt/recon_worker.hpp
)
target_link_libraries(sirius-app PRIVATE sirius::app_core sirius::qt)
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-app)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-app)
endif()

# Runtime DLLs on Windows: the NVIDIA libraries (as for the tests and
# benchmarks) and the Qt libraries/plugins via windeployqt, so the executable
# starts from the build tree without PATH changes. SIRIUS_APP_DEPLOY_QT opts
# out of the Qt part (see ProjectOptions.cmake).
sirius_copy_runtime_dlls(sirius-app)
if(SIRIUS_APP_DEPLOY_QT)
    get_target_property(_qmake_location Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_location}" DIRECTORY)
    find_program(SIRIUS_WINDEPLOYQT windeployqt HINTS "${_qt_bin_dir}")
    if(SIRIUS_WINDEPLOYQT)
        # windeployqt infers debug/release from the executable and then looks
        # for the matching Qt DLLs. A release-only Qt (conda, vcpkg release
        # triplets) has no Qt*d.dll: our Debug executable links its release
        # libraries anyway, so tell windeployqt to deploy those.
        get_target_property(_qt_core_debug Qt${QT_VERSION_MAJOR}::Core IMPORTED_LOCATION_DEBUG)
        if(_qt_core_debug)
            set(_deploy_flavor "$<IF:$<CONFIG:Debug>,--debug,--release>")
        else()
            set(_deploy_flavor "--release")
        endif()
        add_custom_command(TARGET sirius-app POST_BUILD
            COMMAND "${SIRIUS_WINDEPLOYQT}" ${_deploy_flavor} --no-translations --no-compiler-runtime
                    "$<TARGET_FILE:sirius-app>"
            COMMENT "Deploying Qt runtime next to sirius-app"
            VERBATIM)
    else()
        message(WARNING "windeployqt not found; sirius-app needs the Qt bin directory on PATH to run")
    endif()
endif()
