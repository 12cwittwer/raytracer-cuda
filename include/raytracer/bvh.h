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
    hit_record hit(const ray& r, interval ray_t) const {
        hit_record rec;
        rec.hit = false;

        if (left.data == nullptr) {
            printf("BVH left.data is null! left.type=%d\n", (int)left.type);
            rec.hit = false;
            return rec;
        }
        if (right.data == nullptr) {
            printf("BVH right.data is null! right.type=%d\n", (int)right.type);
            rec.hit = false;
            return rec;
        }
        if (!bbox.hit(r, ray_t)) return rec;

        hit_record left_rec = hit_hittable(left, r, ray_t, rec);
        hit_record right_rec = hit_hittable(right, r, interval(ray_t.min, left_rec.hit ? left_rec.t : ray_t.max), rec);

        if (left_rec.hit && (!right_rec.hit || left_rec.t < right_rec.t)) {
            return left_rec;
        } else if (right_rec.hit) {
            return right_rec;
        }

        return rec;
    }
};

#include "hittable_dispatch_impl.h"

#endif
