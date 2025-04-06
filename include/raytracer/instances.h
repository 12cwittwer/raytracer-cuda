#ifndef INSTANCES_H
#define INSTANCES_H

#include "aabb.h"
#include "ray.h"
#include "material.h"
#include "cuda_compat.h"
#include "hittable_dispatch.h"

struct gpu_hittable_list {
    hittable* objects;
    int count;

    __host__ __device__
    aabb bounding_box() const {
        aabb output_box;
        for (int i = 0; i < count; i++) {
            aabb obj_box = ::bounding_box(objects[i]);
            if (i == 0) output_box = obj_box;
            else output_box = aabb(output_box, obj_box);
        }
        return output_box;
    }
};

__host__ __device__
inline hit_record hit_gpu_hittable_list(const gpu_hittable_list& list, const ray& r, interval ray_t, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < list.count; i++) {
        temp_rec = hit_hittable(list.objects[i], r, interval(ray_t.min, closest_so_far), rec);
        if (temp_rec.hit) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return rec;
}

struct translate {
    vec3 offset;
    hittable* object;
    aabb bbox;
    
    __host__ __device__
    translate() {}
    
    __host__ __device__
    translate(hittable* object, const vec3& offset)
    : offset(offset), object(object) {
        bbox = ::bounding_box(*object) + offset;
    }
    
    __host__ __device__
    aabb bounding_box() const {
        return bbox;
    }
};

struct rotate_y {
    hittable* object;
    double sin_theta;
    double cos_theta;
    aabb bbox;
    
    __host__ __device__
    rotate_y() {}
    
    __host__ __device__
    rotate_y(hittable* object, double angle)
    : object(object) {
        auto radians = degrees_to_radians(angle);
        sin_theta = sin(radians);
        cos_theta = cos(radians);
        
        auto b = ::bounding_box(*object);
        point3 min( infinity,  infinity,  infinity);
        point3 max(-infinity, -infinity, -infinity);
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * b.x.max + (1 - i) * b.x.min;
                    auto y = j * b.y.max + (1 - j) * b.y.min;
                    auto z = k * b.z.max + (1 - k) * b.z.min;

                    auto newx = cos_theta * x + sin_theta * z;
                    auto newz = -sin_theta * x + cos_theta * z;
                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fmin(min[c], tester[c]);
                        max[c] = fmax(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    __host__ __device__
    aabb bounding_box() const {
        return bbox;
    }
};

__host__ __device__
inline hit_record hit_translate(const translate& t, const ray& r, interval ray_t, hit_record& rec) {
    ray offset_r(r.origin() - t.offset, r.direction());
    hit_record temp_rec = hit_hittable(*t.object, offset_r, ray_t, rec);

    if (!temp_rec.hit)
        return rec;

    temp_rec.p += t.offset;
    return temp_rec;
}

__host__ __device__
inline hit_record hit_rotate(const rotate_y& rot, const ray& r, interval ray_t, hit_record& rec) {
    auto origin = point3(
        rot.cos_theta * r.origin().x() - rot.sin_theta * r.origin().z(),
        r.origin().y(),
        rot.sin_theta * r.origin().x() + rot.cos_theta * r.origin().z()
    );

    auto direction = vec3(
        rot.cos_theta * r.direction().x() - rot.sin_theta * r.direction().z(),
        r.direction().y(),
        rot.sin_theta * r.direction().x() + rot.cos_theta * r.direction().z()
    );

    ray rotated_r(origin, direction);
    hit_record temp_rec = hit_hittable(*rot.object, rotated_r, ray_t, rec);

    if (!temp_rec.hit)
        return rec;

    temp_rec.p = point3(
        rot.cos_theta * temp_rec.p.x() + rot.sin_theta * temp_rec.p.z(),
        temp_rec.p.y(),
        -rot.sin_theta * temp_rec.p.x() + rot.cos_theta * temp_rec.p.z()
    );

    temp_rec.normal = vec3(
        rot.cos_theta * temp_rec.normal.x() + rot.sin_theta * temp_rec.normal.z(),
        temp_rec.normal.y(),
        -rot.sin_theta * temp_rec.normal.x() + rot.cos_theta * temp_rec.normal.z()
    );

    return temp_rec;
}

#include "hittable_dispatch_impl.h"

#endif