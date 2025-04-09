#pragma once
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "material.h"
#include "cuda_compat.h"

__device__ color ray_color(
    const ray& r,
    int depth,
    const hittable* world,
    color background,
    curandState rng
) {
    if (depth <= 0)
        return color(0, 0, 0);

    if (depth < 50)
        printf("I made it to %d\n", depth);
        
    hit_record rec;
    interval t_range(0.001, infinity);
    rec = hit_hittable(*world, r, t_range, rec);
    if (depth == 49 && rec.hit) {
        printf("hit at t=%.2f, mat type=%d, mat_ptr=%p\n", rec.t, (int)rec.mat_ptr->type, rec.mat_ptr->data);
    }
    if (!rec.hit)
        return background;

    ray scattered;
    color attenuation;
    color emitted = emitted_material(*rec.mat_ptr, rec.u, rec.v, rec.p);
    if (depth == 49) {
        printf("Scattering material type=%d\n", (int)rec.mat_ptr->type);
    }
    if (!scatter_material(*rec.mat_ptr, r, rec, attenuation, scattered, &rng))
        return emitted;

    return emitted + attenuation * ray_color(scattered, depth - 1, world, background, rng);
}