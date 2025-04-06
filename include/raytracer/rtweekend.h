#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <cstdlib> // Will need to change std random to curand for CUDA compatablility

#include "cuda_compat.h"

// Constants

inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline constexpr double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Random Number Utilities (CPU-compatible only)
inline double random_double() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

// Common Headers (these are already CUDA-safe)
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "interval.h"

#endif
