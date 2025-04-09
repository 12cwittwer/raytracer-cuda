#ifndef RENDER_H
#define RENDER_H

#include "camera_data.h"
#include "hittable.h"
#include "material.h"
#include "hittable_dispatch_impl.h"

// CUDA kernel
__global__ void render_kernel(
    const camera_data* cam,
    const hittable* world,
    color* framebuffer
);

// C-linkage for host-callable kernel launch
#ifdef __cplusplus
extern "C" {
#endif

void launch_render_kernel(const camera_data* cam, const hittable* world, color* fb, int image_width, int image_height);

#ifdef __cplusplus
}
#endif

#endif // RENDER_H
