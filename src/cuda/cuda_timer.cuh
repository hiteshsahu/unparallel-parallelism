#pragma once
#include "cuda_check.cuh"

// RAII CUDA event timer.
// Usage:
//   CudaTimer t;
//   t.start();
//   kernel<<<...>>>(...);
//   float ms = t.stop_ms();   // synchronizes and returns elapsed milliseconds
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    // Records stop, synchronizes, and returns elapsed time in milliseconds.
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    CudaTimer(const CudaTimer &)            = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

private:
    cudaEvent_t start_, stop_;
};
