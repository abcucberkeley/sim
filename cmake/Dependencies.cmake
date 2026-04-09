find_package(Eigen3 CONFIG REQUIRED)
find_package(TIFF REQUIRED)
find_package(OpenMP REQUIRED)

if (SIRIUS_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

if (SIRIUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(SIRIUS_ENABLE_TESTS)
    find_package(Catch2 3 CONFIG REQUIRED)
endif()
