#ifndef QUAD_H
#define QUAD_H

#include "aabb.h"
#include "ray.h"
#include "material.h"
#include "cuda_compat.h"

struct quad {
    point3 Q;
    vec3 u, v;
    vec3 w;
    vec3 normal;
    double D;
    material* mat_ptr;
    aabb bbox;

    __host__ __device__
    quad() {}

    __host__ __device__
    quad(const point3& Q, const vec3& u, const vec3& v, material* mat)
        : Q(Q), u(u), v(v), mat_ptr(mat)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);

        set_bounding_box();
    }

    __host__ __device__
    void set_bounding_box() {
        auto min = Q;
        auto max = Q + u + v;
        bbox = aabb(min, max);
        bbox = aabb(bbox, aabb(Q + u, Q + v));
    }

    __host__ __device__
    aabb bounding_box() const {
        return bbox;
    }
};

__host__ __device__ inline hit_record hit_quad(
    const quad& quad, const ray& r, interval ray_t, hit_record rec
) {
    if (quad.mat_ptr == nullptr) {
        printf("Quad material pointer is null");
    }
    auto denom = dot(quad.normal, r.direction());
    if (fabs(denom) < 1e-8)
        return rec;

    auto t = (quad.D - dot(quad.normal, r.origin())) / denom;
    if (!ray_t.contains(t))
        return rec;

    point3 p = r.at(t);
    vec3 planar_vec = p - quad.Q;
    auto alpha = dot(cross(planar_vec, quad.v), quad.normal) / dot(cross(quad.u, quad.v), quad.normal);
    auto beta  = dot(cross(quad.u, planar_vec), quad.normal) / dot(cross(quad.u, quad.v), quad.normal);

    if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
        return rec;

    rec.t = t;
    rec.p = p;
    rec.mat_ptr = quad.mat_ptr;
    rec.u = alpha;
    rec.v = beta;
    rec.set_face_normal(r, quad.normal);
    rec.hit = true;

    return rec;
}


#endif
