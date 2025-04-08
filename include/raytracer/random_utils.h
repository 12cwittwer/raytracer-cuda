// random_utils.h (CUDA-friendly version)
#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <curand_kernel.h>
#include "vec3.h"

// ==================== HOST-SIDE RANDOM FUNCTIONS ====================
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

CUDA_HOST_DEVICE inline double random_double_host() {
    // Returns a random real in [0,1) using std::rand (host only)
    return rand() / (RAND_MAX + 1.0);
}

CUDA_HOST_DEVICE inline double random_double_host(double min, double max) {
    return min + (max - min) * random_double_host();
}

CUDA_HOST_DEVICE inline vec3 random_vec3_host() {
    return vec3(
        random_double_host(),
        random_double_host(),
        random_double_host()
    );
}

CUDA_HOST_DEVICE inline vec3 random_vec3_host(double min, double max) {
    return vec3(
        random_double_host(min, max),
        random_double_host(min, max),
        random_double_host(min, max)
    );
}

CUDA_HOST_DEVICE inline vec3 random_in_unit_disk_host() {
    while (true) {
        auto p = vec3(cpu_random_double()*2 - 1, cpu_random_double()*2 - 1, 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

CUDA_HOST_DEVICE inline vec3 random_in_unit_sphere_host() {
    while (true) {
        auto p = random_vec3_host(-1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

CUDA_HOST_DEVICE inline vec3 random_unit_vector_host() {
    return unit_vector(random_in_unit_sphere_host());
}



// ==================== DEVICE-SIDE RANDOM FUNCTIONS ====================
__device__ inline double random_double(curandState* state) {
    return curand_uniform_double(state);
}

__device__ inline double random_double(double min, double max, curandState* state) {
    return min + (max - min) * random_double(state);
}

__device__ inline vec3 random_vec3(curandState* state) {
    return vec3(
        random_double(state),
        random_double(state),
        random_double(state)
    );
}

__device__ inline vec3 random_vec3(double min, double max, curandState* state) {
    return vec3(
        random_double(min, max, state),
        random_double(min, max, state),
        random_double(min, max, state)
    );
}

__device__ inline vec3 random_in_unit_sphere(curandState* state) {
    while (true) {
        auto p = random_vec3(-1, 1, state);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState* state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ inline vec3 random_in_hemisphere(const vec3& normal, curandState* state) {
    vec3 in_unit_sphere = random_in_unit_sphere(state);
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__device__ inline vec3 random_in_unit_disk(curandState* state) {
    while (true) {
        auto p = vec3(random_double(state)*2 - 1, random_double(state)*2 - 1, 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

#endif // RANDOM_UTILS_H