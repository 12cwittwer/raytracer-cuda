#ifndef SPHERE_GPU_H
#define SPHERE_GPU_H

#include "rtweekend.h"
#include "ray.h"
#include "cuda_compat.h"

struct gpu_sphere {
    point3 center;
    double radius;
    material* mat_ptr;

    __host__ __device__
    gpu_sphere() : center(), radius(0) {}

    __host__ __device__
    gpu_sphere(const point3& c, double r, material* mat_ptr) : center(c), radius(r), mat_ptr(mat_ptr) {}

    __host__ __device__
    aabb bounding_box() const {
        vec3 rvec(radius, radius, radius);
        return aabb(center - rvec, center + rvec);
    }
};

__host__ __device__ inline hit_record hit_sphere(const gpu_sphere& s, const ray& r, interval ray_t, hit_record& rec) {
    if (s.mat_ptr == nullptr) {
        printf("Sphere material pointer is null");
    }

    vec3 oc = r.origin() - s.center;
    auto a = dot(r.direction(), r.direction());
    auto half_b = dot(oc, r.direction());
    auto c = dot(oc, oc) - s.radius * s.radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0) return hit_record{};
    auto sqrt_d = sqrt(discriminant);

    auto root = (-half_b - sqrt_d) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrt_d) / a;
        if (!ray_t.surrounds(root))
            return hit_record{};
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - s.center) / s.radius;
    rec.set_face_normal(r, outward_normal);
    rec.hit = true;
    rec.mat_ptr = s.mat_ptr;

    return rec;
}


#endif