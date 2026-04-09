add_library(sirius STATIC)

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
            ${PROJECT_SOURCE_DIR}/include/sirius/tiff_io.hpp
    PRIVATE
        tiff_io.cpp
)

target_link_libraries(sirius
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        TIFF::TIFF
        OpenMP::OpenMP_CXX
)

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius)
endif()
target_enable_sanitizers(sirius) 