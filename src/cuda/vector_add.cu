#include <iostream>
#include <vector>
#include "device_vector.cuh"
#include "vector_add.cuh"


int main() {

    const int N = 1 << 20; // 1M elements

    // ✨ Initialize Host Vectors
    std::vector<int> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 📤 Convert & upload Host Vector to Device vector
    DeviceVector<int> d_a(N), d_b(N), d_c(N);
    d_a.upload(h_a);
    d_b.upload(h_b);

    // ⏳ Execute on GPU
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    vectorAdd<<<BLOCKS, THREADS>>>(d_a.data(), d_b.data(), d_c.data(), N);

    // Validate Errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //  📤 Download result from device to host
    d_c.download(h_c);

    //📜 Print
    std::cout << "First result: " << h_a[0] << " + " << h_b[0] << " = " << h_c[0] << "\n";
    std::cout << "Last  result: " << h_a[N-1] << " + " << h_b[N-1] << " = " << h_c[N-1] << "\n";

    return 0;
}
