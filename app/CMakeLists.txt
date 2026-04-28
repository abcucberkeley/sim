add_executable(SiriusGUI
    main.cpp
)

target_link_libraries(SiriusGUI PRIVATE
    sirius
    imgui::imgui
    SDL3::SDL3
    OpenGL::GL
)

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(SiriusGUI)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(SiriusGUI)
endif()