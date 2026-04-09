add_executable(SiriusGUI
    main.cpp
)

target_link_libraries(SiriusGUI PRIVATE
    sirius
    imgui::imgui
    SDL3::SDL3
    OpenGL::GL
)