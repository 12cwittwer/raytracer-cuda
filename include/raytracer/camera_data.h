#ifndef CAMERA_DATA_H
#define CAMERA_DATA_H

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "random_utils.h"

struct camera_data {
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;
    vec3 u, v, w;
    double defocus_angle;
    double pixel_samples_scale;
    color background;
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
};

__device__ inline vec3 sample_square(curandState* rng) {
    return vec3(random_double(rng) - 0.5, random_double(rng) - 0.5, 0);
}

__device__ inline point3 defocus_disk_sample(const camera_data* cam, curandState* rng) {
    auto p = random_in_unit_disk(rng);
    return cam->center + (p[0] * cam->defocus_disk_u) + (p[1] * cam->defocus_disk_v);
}

__device__ inline ray get_ray(const camera_data* cam, int i, int j, curandState* rng) {
    vec3 offset = sample_square(rng);
    auto pixel_sample = cam->pixel00_loc 
                        + ((i + offset.x()) * cam->pixel_delta_u)
                        + ((j + offset.y()) * cam->pixel_delta_v);
    
    auto ray_origin = (cam->defocus_angle <= 0) ? cam->center : defocus_disk_sample(cam, rng);
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}






#endif
