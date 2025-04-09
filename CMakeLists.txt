cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX CUDA)

enable_language(CUDA)

# Optional: set CUDA version compatibility
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable with source files
add_executable(raytracer
    src/cpu/main.cpp
    src/gpu/render.cu
    src/common/utils.inl
)

# Include paths
target_include_directories(raytracer PRIVATE include/ /usr/local/cuda/include)

# CUDA settings
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53"  # or "72" for Jetson Xavier, "53" for Nano
)
