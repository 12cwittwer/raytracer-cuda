#ifndef HITTABLE_DISPATCH_IMPL_H
#define HITTABLE_DISPATCH_IMPL_H

#include "hittable_dispatch.h"
#include "hittable.h"
#include "sphere_gpu.h"
#include "bvh.h"  // Full definition of bvh_node included here
#include "instances.h"

__host__ __device__ inline
hit_record hit_hittable(const hittable& h, const ray& r, interval ray_t, hit_record& rec) {
    if (h.data == nullptr) {
        printf("hit_hittable: null data pointer! type=%d\n", (int)h.type);
        rec.hit = false;
        return rec;
    }
    printf("hit_hittable type=%d data=%p\n", (int)h.type, h.data);
    switch (h.type) {
        case hittable_type::sphere:
            return hit_sphere(*reinterpret_cast<const gpu_sphere*>(h.data), r, ray_t, rec);
        case hittable_type::bvh_node:
            return reinterpret_cast<const bvh_node*>(h.data)->hit(r, ray_t);
        case hittable_type::quad:
            return hit_quad(*reinterpret_cast<const quad*>(h.data), r, ray_t, rec);
        case hittable_type::translate:
            return hit_translate(*reinterpret_cast<const translate*>(h.data), r, ray_t, rec);
        case hittable_type::rotate_y:
            return hit_rotate(*reinterpret_cast<const rotate_y*>(h.data), r, ray_t, rec);
        case hittable_type::hittable_list:
            return hit_gpu_hittable_list(*reinterpret_cast<const gpu_hittable_list*>(h.data), r, ray_t, rec);        
        default:
            return hit_record{};
    }
}

__host__ __device__ inline
aabb bounding_box(const hittable& h) {
    switch (h.type) {
        case hittable_type::sphere:
            return reinterpret_cast<const gpu_sphere*>(h.data)->bounding_box();
        case hittable_type::bvh_node:
            return reinterpret_cast<const bvh_node*>(h.data)->bbox;
        case hittable_type::quad:
            return reinterpret_cast<const quad*>(h.data)->bounding_box();
        case hittable_type::translate:
            return reinterpret_cast<const translate*>(h.data)->bounding_box();
        case hittable_type::rotate_y:
            return reinterpret_cast<const rotate_y*>(h.data)->bounding_box();
        case hittable_type::hittable_list:
            return reinterpret_cast<const gpu_hittable_list*>(h.data)->bounding_box();        
        default:
            return aabb();
    }
}

#endif
