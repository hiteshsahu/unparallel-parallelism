#pragma once
#include <vector>
#include "device_vector.cuh"
#include "cuda_check.cuh"


// Grid-stride loop handles any N, not just N <= 1024.
__global__ void vectorAdd(const int *a, const int *b, int *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}
