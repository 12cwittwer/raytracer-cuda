#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "aabb.h"
#include "hittable_dispatch_impl.h"

// Flattened version for GPU-compatibility
struct hittable_list {
    hittable* objects;
    int count;

    __host__ __device__
    void hit(const ray& r, interval ray_t) const {
        hit_record final_rec;
        final_rec.hit = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < count; ++i) {
            hit_hittable(objects[i], r, interval(ray_t.min, closest_so_far), final_rec);
            if (final_rec.hit) {
                closest_so_far = final_rec.t;
            }
        }
    }

    __host__ __device__
    aabb bounding_box() const {
        if (count == 0) return aabb();

        aabb output_box = ::bounding_box(objects[0]);

        for (int i = 1; i < count; ++i) {
            aabb b = ::bounding_box(objects[i]);
            output_box = surrounding_box(output_box, b);
        }

        return output_box;
    }
};

#endif
