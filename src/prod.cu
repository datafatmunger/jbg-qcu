// #include <iostream>
// #include <cuda_runtime.h>

// #define BLOCK_SIZE 16

// // CUDA kernel for matrix-vector multiplication
// __global__ void matVecMulKernel(float *A, float *x, float *y, int rows, int cols) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     if (row < rows) {
//         float sum = 0.0f;
//         for (int col = 0; col < cols; ++col) {
//             sum += A[row * cols + col] * x[col];
//         }
//         y[row] = sum;
//     }
// }

// void matVecMul(float *h_A, float *h_x, float *h_y, int rows, int cols) {
//     float *d_A, *d_x, *d_y;

//     // Allocate memory on the device
//     cudaMalloc((void**)&d_A, rows * cols * sizeof(float));
//     cudaMalloc((void**)&d_x, cols * sizeof(float));
//     cudaMalloc((void**)&d_y, rows * sizeof(float));

//     // Copy data from host to device
//     cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);

//     // Define the grid and block dimensions
//     dim3 dimBlock(1, BLOCK_SIZE);
//     dim3 dimGrid(1, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

//     // Launch the kernel
//     matVecMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, rows, cols);

//     // Copy the result back to the host
//     cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

//     // Clean up
//     cudaFree(d_A);
//     cudaFree(d_x);
//     cudaFree(d_y);
// }

// int main() {
//     int rows = 4;
//     int cols = 4;

//     // Host input matrices
//     float h_A[] = {
//         1,  0,  1,  0,
//         0,  1,  0,  1,
//         1,  0, -1,  0,
//         0,  1,  0, -1};
//     float h_x[] = {1, 0, 0, 0};
//     float h_y[4];

//     // Matrix-vector multiplication
//     matVecMul(h_A, h_x, h_y, rows, cols);

//     // Print the result
//     for (int i = 0; i < rows; ++i) {
//         std::cout << h_y[i] << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }

#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define BLOCK_SIZE 16

// CUDA kernel for complex matrix-vector multiplication
__global__ void matVecMulKernel(cuFloatComplex *A, cuFloatComplex *x, cuFloatComplex *y, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int col = 0; col < cols; ++col) {
            sum = cuCaddf(sum, cuCmulf(A[row * cols + col], x[col]));
        }
        y[row] = sum;
    }
}

void matVecMul(cuFloatComplex *h_A, cuFloatComplex *h_x, cuFloatComplex *h_y, int rows, int cols) {
    cuFloatComplex *d_A, *d_x, *d_y;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, rows * cols * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_x, cols * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_y, rows * sizeof(cuFloatComplex));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 dimBlock(1, BLOCK_SIZE);
    dim3 dimGrid(1, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matVecMulKernel<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, rows, cols);

    // Copy the result back to the host
    cudaMemcpy(h_y, d_y, rows * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int rows = 4;
    int cols = 4;

    // Host input matrices
    cuFloatComplex h_A[] = {
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0)
    };
    cuFloatComplex h_x[] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0)};
    cuFloatComplex h_y[4];

    // Matrix-vector multiplication
    matVecMul(h_A, h_x, h_y, rows, cols);

    // Print the result
    for (int i = 0; i < rows; ++i) {
        //std::cout << "(" << cuCrealf(h_y[i]) << ", " << cuCimagf(h_y[i]) << ") ";
        std::cout << cuCrealf(h_y[i]) << " + " << cuCimagf(h_y[i]) << "i ";
    }
    std::cout << std::endl;

    return 0;
}
