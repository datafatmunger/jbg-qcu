#include "qsim.h"

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

__global__ void tensorProductKernel(cuFloatComplex* d_A, cuFloatComplex* d_B, cuFloatComplex* d_C, int aRows, int aCols, int bRows, int bCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int totalRows = aRows * bRows;
    int totalCols = aCols * bCols;

    if (i < totalRows && j < totalCols) {
        int rowA = i / bRows;
        int colA = j / bCols;
        int rowB = i % bRows;
        int colB = j % bCols;
        d_C[i * totalCols + j] = cuCmulf(d_A[rowA * aCols + colA], d_B[rowB * bCols + colB]);
    }
}

void tensorProduct(cuFloatComplex* h_A, cuFloatComplex* h_B, cuFloatComplex* h_C, int aRows, int aCols, int bRows, int bCols) {
    int aSize = aRows * aCols * sizeof(cuFloatComplex);
    int bSize = bRows * bCols * sizeof(cuFloatComplex);
    int cSize = aRows * bRows * aCols * bCols * sizeof(cuFloatComplex);

    cuFloatComplex* d_A;
    cuFloatComplex* d_B;
    cuFloatComplex* d_C;

    cudaMalloc((void**)&d_A, aSize);
    cudaMalloc((void**)&d_B, bSize);
    cudaMalloc((void**)&d_C, cSize);

    cudaMemcpy(d_A, h_A, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((aCols * bCols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (aRows * bRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tensorProductKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, aRows, aCols, bRows, bCols);

    cudaMemcpy(h_C, d_C, cSize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}