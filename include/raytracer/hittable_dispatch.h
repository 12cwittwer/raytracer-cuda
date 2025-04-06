#ifndef HITTABLE_DISPATCH_H
#define HITTABLE_DISPATCH_H

#include "interval.h"
#include "aabb.h"
#include "ray.h"
#include "hittable.h"

struct bvh_node; // Forward declare only

__host__ __device__
hit_record hit_hittable(const hittable& h, const ray& r, interval ray_t, hit_record& rec);

__host__ __device__
aabb bounding_box(const hittable& h);

#endif
