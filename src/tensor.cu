#include <cuda_runtime.h>
#include <iostream>

__global__ void tensorProductKernel(float* d_A, float* d_B, float* d_C, int aRows, int aCols, int bRows, int bCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int totalRows = aRows * bRows;
    int totalCols = aCols * bCols;

    if (i < totalRows && j < totalCols) {
        int rowA = i / bRows;
        int colA = j / bCols;
        int rowB = i % bRows;
        int colB = j % bCols;
        d_C[i * totalCols + j] = d_A[rowA * aCols + colA] * d_B[rowB * bCols + colB];
    }
}

void tensorProduct(float* h_A, float* h_B, float* h_C, int aRows, int aCols, int bRows, int bCols) {
    int aSize = aRows * aCols * sizeof(float);
    int bSize = bRows * bCols * sizeof(float);
    int cSize = aRows * bRows * aCols * bCols * sizeof(float);

    float* d_A;
    float* d_B;
    float* d_C;

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

int main() {
    const int aRows = 2;
    const int aCols = 2;
    const int bRows = 2;
    const int bCols = 2;

    float h_A[aRows * aCols] = {1, 1, 1, -1};
    float h_B[bRows * bCols] = {1, 0, 0, 1};
    float h_C[aRows * bRows * aCols * bCols];

    tensorProduct(h_A, h_B, h_C, aRows, aCols, bRows, bCols);

    for (int i = 0; i < aRows * bRows; ++i) {
        for (int j = 0; j < aCols * bCols; ++j) {
            std::cout << h_C[i * aCols * bCols + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>

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

int main() {
    const int aRows = 2;
    const int aCols = 2;
    const int bRows = 2;
    const int bCols = 2;

    cuFloatComplex h_A[aRows * aCols] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(1, 0), make_cuFloatComplex(1, 0), make_cuFloatComplex(-1, 0)};
    cuFloatComplex h_B[bRows * bCols] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0)};
    cuFloatComplex h_C[aRows * bRows * aCols * bCols];

    tensorProduct(h_A, h_B, h_C, aRows, aCols, bRows, bCols);

    for (int i = 0; i < aRows * bRows; ++i) {
        for (int j = 0; j < aCols * bCols; ++j) {
            std::cout << cuCrealf(h_C[i * aCols * bCols + j]) << "+" << cuCimagf(h_C[i * aCols * bCols + j]) << "i ";
        }
        std::cout << "\n";
    }

    return 0;
}
