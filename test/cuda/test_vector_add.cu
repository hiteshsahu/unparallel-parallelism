#include <iostream>
#include <vector>
#include "vector_add.cuh"

int main() {
    const int N = 1 << 20;

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
    vectorAdd<<<BLOCKS, THREADS>>>(d_a.data(), d_b.data(), d_c.data(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_c.download(h_c);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "FAIL — mismatch at index " << i
                      << ": expected " << (h_a[i] + h_b[i])
                      << ", got " << h_c[i] << "\n";
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "PASS" : "FAIL")
              << " — vector_add over " << N << " elements\n";

    return ok ? 0 : 1;
}
