// vector_add.cu
#include <iostream>

// 🧮 GPU kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int ARRAY_SIZE = 1000;
    int a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

    // ✨ Initialize input arrays
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    size_t size = ARRAY_SIZE * sizeof(int);

    // 📥 Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 📄 Copy data to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // ⏳Launch kernel: 1 block, N threads
    vectorAdd<<<1, N>>>(d_a, d_b, d_c, ARRAY_SIZE);

    // 📤 Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 📜 Print results
    std::cout << "Vector Addition Result:\n";
    for (int i = 0; i < N; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";

    // ♻️ Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
