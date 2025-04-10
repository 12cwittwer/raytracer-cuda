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
    color* framebuffer
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 && y == 0) {
        printf("Kernel launched\n");
    }

    if (x >= cam->image_width || y >= cam->image_height) return;

    if (x == 0 && y == 0) {
        printf("Camera Dimensions Width: %d Height: %d\n", cam->image_width, cam->image_height);
    }

    if (x == 0 && y == 0) {
        printf("First hittable is %d\n", (int)world->type);
        printf("World data is %p\n", world->data);
    }

    printf("World first hittable is %d\n", (int)world->type);
    printf("World data is %p\n", world->data);

    int pixel_index = y * cam->image_width + x;
    curandState rng;
    curand_init(1984 + pixel_index, 0, 0, &rng);

    color pixel_color(0, 0, 0);
    for (int s = 0; s < cam->samples_per_pixel; ++s) {
        if (x == 0 && y == 0) {
            printf("cam=%p\n", cam);
            printf("cam->image_width=%d\n", cam->image_width);
            printf("cam->samples_per_pixel=%d\n", cam->samples_per_pixel);
            printf("cam->background=(%f, %f, %f)\n",
                cam->background.x(), cam->background.y(), cam->background.z());
        }
        ray r = get_ray(cam, x, y, &rng);
        if (world->data == nullptr) {
            if (x == 0 && y == 0) printf("CRASH PREVENTED: world->data was null\n");
            return;
        }
        pixel_color += ray_color(r, cam->max_depth, world, cam->background, rng);
    }

    framebuffer[pixel_index] = pixel_color / cam->samples_per_pixel;
}

#include "raytracer/cuda_utils.h"  // for CUDA_CHECK

void launch_render_kernel(const camera_data* cam, const hittable* world, color* fb, int image_width, int image_height) {
    const dim3 threads_per_block(8, 8);
    const dim3 num_blocks(
        (image_width + threads_per_block.x - 1) / threads_per_block.x,
        (image_height + threads_per_block.y - 1) / threads_per_block.y
    );

    // Launch kernel
    render_kernel<<<num_blocks, threads_per_block>>>(cam, world, fb);

    // Check for immediate kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Ensure kernel is finished before moving on
    CUDA_CHECK(cudaDeviceSynchronize());
}
