cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX CUDA)

# Find MPI
find_package(MPI REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Set Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# CUDA specific flags
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Debug options
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0 -g -lineinfo")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Sources
add_executable(raytracer
    src/cpu/main.cpp
    src/gpu/render.cu
    src/common/utils.inl
)

# Include paths
target_include_directories(raytracer PRIVATE
    include/
    ${CUDA_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
)

# Link Libraries
target_link_libraries(raytracer PRIVATE
    ${CUDA_LIBRARIES}
    MPI::MPI_CXX
)

# CUDA Properties
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53"  # Adjust for Jetson (53 = Maxwell, 72 = Xavier, 87 = Orin)
)
