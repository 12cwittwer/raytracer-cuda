enable_language(CUDA)

add_executable(raytracer
    src/cpu/main.cpp
    src/gpu/render.cu
    src/common/utils.inl
)

target_include_directories(raytracer PRIVATE include/)
set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
