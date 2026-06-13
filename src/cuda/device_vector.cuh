#pragma once
#include <vector>
#include "cuda_check.cuh"

// RAII wrapper around a device buffer.
// Allocates on construction, frees on destruction — no manual cudaFree needed.
template <typename T>
class DeviceVector {
public:
    explicit DeviceVector(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    ~DeviceVector() { cudaFree(ptr_); }

    void upload(const std::vector<T> &host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(),
                              count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(std::vector<T> &host) const {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_,
                              count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T       *data()       { return ptr_; }
    const T *data() const { return ptr_; }
    size_t   size() const { return count_; }

    DeviceVector(const DeviceVector &)            = delete;
    DeviceVector &operator=(const DeviceVector &) = delete;

private:
    T      *ptr_   = nullptr;
    size_t  count_ = 0;
};
