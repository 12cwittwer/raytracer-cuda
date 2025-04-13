#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "raytracer/vec3.h"
#include "raytracer/ray.h"
#include "raytracer/interval.h"
#include "raytracer/color.h"

#include "raytracer/camera_data.h"
#include "raytracer/material.h"
#include "raytracer/hittable.h"
#include "raytracer/hittable_dispatch.h"
#include "raytracer/hittable_dispatch_impl.h"
#include "raytracer/bvh.h"
#include "raytracer/sphere_gpu.h"
#include "raytracer/quad.h"
#include "raytracer/instances.h"
#include "raytracer/device_ray_color.h"
#include "raytracer/random_utils.h"

#include "raytracer/render.h"


__global__ void render_kernel(
    const camera_data* cam,
    const hittable* world,
    color* framebuffer,
    int row
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = row;

    if (x >= cam->image_width) return;

    int pixel_index = x;
    curandState rng;
    int seed = 1984 + row * cam->image_width + x;
    curand_init(seed, 0, 0, &rng);

    color pixel_color(0, 0, 0);
    for (int s = 0; s < cam->samples_per_pixel; ++s) {
        ray r = get_ray(cam, x, y, &rng);
        pixel_color += ray_color(r, cam->max_depth, world, cam->background, rng);
    }
    pixel_color /= cam->samples_per_pixel;
    framebuffer[pixel_index] = pixel_color;
}

#include "raytracer/cuda_utils.h"  // for CUDA_CHECK

void launch_render_kernel(const camera_data* cam, const hittable* world, color* fb, int image_width, int image_height, int row) {
    const dim3 threads_per_block(512, 1);
    const dim3 num_blocks((image_width + threads_per_block.x - 1) / threads_per_block.x, 1);

    // Launch kernel
    // Try rendering the first pixel
    // render_kernel<<<1, 1>>>(cam, world, fb);
    render_kernel<<<num_blocks, threads_per_block>>>(cam, world, fb, row);

    // Check for immediate kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Ensure kernel is finished before moving on
    // CUDA_CHECK(cudaDeviceSynchronize());
}
