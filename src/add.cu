// add.cu

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function to add two numbers
__global__ void add(int* a, int* b, int* c) {
    *c = *a + *b;
}

int main() {
    // Host variables
    int a = 3;
    int b = 5;
    int c = 0;

    // Device variables
    int *d_a, *d_b, *d_c;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    // Copy host variables to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with one block and one thread
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Copy the result back to the host
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result: " << c << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
