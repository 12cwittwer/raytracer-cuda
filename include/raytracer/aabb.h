#ifndef AABB_H
#define AABB_H

#include "interval.h"
#include "ray.h"

class aabb {
  public:
    interval x, y, z;

    // Default constructor makes an empty box
    __host__ __device__
    aabb() {}

    __host__ __device__
    aabb(const interval& x, const interval& y, const interval& z)
      : x(x), y(y), z(z) {
        pad_to_minimums();
      }

    __host__ __device__
    aabb(const point3& a, const point3& b) {
        x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);

        pad_to_minimums();
    }

    __host__ __device__
    aabb(const aabb& bbox0, const aabb& bbox1) {
        x = interval(bbox0.x, bbox1.x);
        y = interval(bbox0.y, bbox1.y);
        z = interval(bbox0.z, bbox1.z);
    }

    __host__ __device__
    const interval& axis_interval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    __host__ __device__
    bool hit(const ray& r, interval ray_t) const {
        const point3& ray_orig = r.origin();
        const vec3&   ray_dir  = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axis_interval(axis);
            const double adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

    __host__ __device__
    point3 min() const { return point3(x.min, y.min, z.min); }

    __host__ __device__
    point3 max() const { return point3(x.max, y.max, z.max); }

    __host__ __device__
    void pad_to_minimums() {
        double delta = 0.0001;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

// Surrounding box utility (GPU-safe)
__host__ __device__
inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    return aabb(
        interval(fmin(box0.x.min, box1.x.min), fmax(box0.x.max, box1.x.max)),
        interval(fmin(box0.y.min, box1.y.min), fmax(box0.y.max, box1.y.max)),
        interval(fmin(box0.z.min, box1.z.min), fmax(box0.z.max, box1.z.max))
    );
}

__host__ __device__
inline aabb operator+(const aabb& bbox, const vec3& offset) {
    return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__host__ __device__
inline aabb operator+(const vec3& offset, const aabb& bbox) {
    return bbox + offset;
}

#endif
