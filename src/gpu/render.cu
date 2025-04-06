#include <cuda_runtime.h>
#include "raytracer/ray.h"

__global__ void render_kernel() {
    // Empty kernel stub
}

void launch_render_kernel() {
    render_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}