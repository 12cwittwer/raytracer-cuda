#pragma once
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "material.h"
#include "cuda_compat.h"

__device__ color ray_color(
    ray r,
    int depth,
    const hittable* world,
    color background,
    curandState& rng
) {
    color result(0, 0, 0);
    color attenuation(1, 1, 1);

    for (int i = 0; i < depth; i++) {
        hit_record rec = {};
        rec.hit = false;

        hit_hittable(*world, r, interval(0.001, 1.0e30), rec);

        if (!rec.hit) {
            result += attenuation * background;
            break;
        }

        color emitted = emitted_material(*rec.mat_ptr, rec.u, rec.v, rec.p);

        ray scattered;
        color temp_attenuation;

        bool scattered_success = scatter_material(*rec.mat_ptr, r, rec, temp_attenuation, scattered, &rng);

        result += attenuation * emitted;

        if (!scattered_success) {
            break;
        }

        attenuation *= temp_attenuation;
        r = scattered;
    }

    return result;
}
