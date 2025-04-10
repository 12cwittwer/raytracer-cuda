#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " — " << cudaGetErrorString(code) << std::endl;
        exit(code);
    }
}
