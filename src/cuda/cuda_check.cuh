#pragma once
#include <cstdlib>
#include <iostream>

/**
 * Checks error in cuda calls
 **/
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " -- " << cudaGetErrorString(err) << "\n";  \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)
