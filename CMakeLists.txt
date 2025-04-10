cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

# Enable relaxed constexpr for CUDA
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# === CUDA Debug Flags for cuda-gdb ===
set(CMAKE_BUILD_TYPE Debug)

# Add device-side debug info, host-side debug info, and line info for cuda-gdb
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0 -g -lineinfo")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Sources
add_executable(raytracer
    src/cpu/main.cpp
    src/gpu/render.cu
    src/common/utils.inl
)

target_include_directories(raytracer PRIVATE include/)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53"  # Update to match your GPU
)
