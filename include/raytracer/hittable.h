#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"
#include "aabb.h"
#include "bvh.h"

struct material;

// -------------------- hit_record --------------------

struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool hit;
    bool front_face;
    const material* mat_ptr = nullptr;
    
    double u = 0;
    double v = 0;
    
    __host__ __device__
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// -------------------- hittable_type & interface --------------------

enum class hittable_type {
    sphere = 0,
    bvh_node = 1,
    quad = 2,
    translate = 3,
    rotate_y = 4,
    hittable_list = 5
};

struct hittable {
    hittable_type type;
    void* data;
};

#include "sphere_gpu.h"
#include "quad.h"

#endif
