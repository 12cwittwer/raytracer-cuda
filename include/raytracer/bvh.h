#ifndef BVH_H
#define BVH_H

#include "hittable.h"
#include "aabb.h"
#include "ray.h"
#include "hittable_dispatch.h"

struct bvh_node {
    hittable left;
    hittable right;
    aabb bbox;

    __host__ __device__
    void hit(const ray& r, interval ray_t, hit_record& rec) const {
        rec.hit = false;  // Always initialize

        if (!bbox.hit(r, ray_t))
            return;
    
        hit_record temp_rec;
    
        if (left.data)
            hit_hittable(left, r, ray_t, temp_rec);
    
        if (temp_rec.hit) {
            rec = temp_rec;
            ray_t.max = temp_rec.t;  // Shrink interval for early out
        }
    
        temp_rec = hit_record{};
    
        if (right.data)
            hit_hittable(right, r, ray_t, temp_rec);  // FIXED: right not left
    
        if (temp_rec.hit)
            rec = temp_rec;
    }
};

#include "hittable_dispatch_impl.h"

#endif
