cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(CUDA QUIET)

if (CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found - using nvcc at ${CUDA_NVCC_EXECUTABLE}")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

    set(CUDA_SOURCES
        src/gpu/render.cu
    )

else()
    message(STATUS "CUDA not found - building CPU-only version")

    set(CUDA_SOURCES "")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0 -g -lineinfo")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

add_executable(raytracer
    src/cpu/main.cpp
    ${CUDA_SOURCES}
    src/common/utils.inl
)

target_include_directories(raytracer PRIVATE
    include/
    ${MPI_INCLUDE_PATH}
)

target_link_libraries(raytracer PRIVATE
    MPI::MPI_CXX
)

if (CUDA_FOUND)
    target_include_directories(raytracer PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES})
    set_target_properties(raytracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "53"
    )
endif()
