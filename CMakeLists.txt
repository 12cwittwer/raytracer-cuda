cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)


# Enable C++17 for both C++ and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Add relaxed constexpr for device-side compatibility (fixes fabs/warnings)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Executable and source files
add_executable(raytracer
    src/cpu/main.cpp
    src/gpu/render.cu
    src/common/utils.inl
)

# Include paths
target_include_directories(raytracer PRIVATE include/)

# CUDA-specific settings
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53"  # or set e.g. 72 for Jetson Xavier, 53 for Nano
)
