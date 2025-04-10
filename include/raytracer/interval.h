#ifndef INTERVAL_H
#define INTERVAL_H

#include "rtweekend.h"

class interval {
  public:
    double min, max;

    __host__ __device__ interval()
        : min(+infinity), max(-infinity) {} // Default = empty

    __host__ __device__ interval(double min, double max)
        : min(min), max(max) {
            printf("Interval successfully called\n");
        }

    __host__ __device__ interval(const interval& a, const interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __host__ __device__ double size() const {
        return max - min;
    }

    __host__ __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }
    
    __host__ __device__
    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__
    interval expand(double delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ static inline interval empty() {
        return interval(+infinity, -infinity);
    }

    __host__ __device__ static inline interval universe() {
        return interval(-infinity, +infinity);
    }
};

__host__ __device__
inline interval operator+(const interval& ival, double displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__
inline interval operator+(double displacement, const interval& ival) {
    return ival + displacement;
}

#endif
