#include <iostream>
#include <vector>
#include "device_vector.cuh"
#include "vector_add.cuh"
#include "cuda_timer.cuh"

int main() {
    const int N = 1 << 20; // 1M elements

    std::vector<int> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    DeviceVector<int> d_a(N), d_b(N), d_c(N);
    d_a.upload(h_a);
    d_b.upload(h_b);

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    CudaTimer timer;
    timer.start();
    vectorAdd<<<BLOCKS, THREADS>>>(d_a.data(), d_b.data(), d_c.data(), N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stop_ms(); // synchronizes internally — no separate cudaDeviceSynchronize needed

    d_c.download(h_c);

    // Bandwidth: kernel reads 2 arrays and writes 1 → 3 * N * sizeof(int) bytes
    float bytes     = 3.f * N * sizeof(int);
    float bandwidth = (bytes / 1e9f) / (ms / 1e3f); // GB/s

    std::cout << "First result : " << h_a[0]   << " + " << h_b[0]   << " = " << h_c[0]   << "\n";
    std::cout << "Last  result : " << h_a[N-1] << " + " << h_b[N-1] << " = " << h_c[N-1] << "\n";
    std::cout << "Kernel time  : " << ms        << " ms\n";
    std::cout << "Bandwidth    : " << bandwidth  << " GB/s\n";

    return 0;
}
