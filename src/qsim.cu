#include "qsim.h"

#define BLOCK_SIZE 16

cuFloatComplex I::gateMatrix[H::matrixSize] = {
    make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0),
    make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0)
};

cuFloatComplex H::gateMatrix[H::matrixSize] = {
    make_cuFloatComplex(1, 0), make_cuFloatComplex(1, 0),
    make_cuFloatComplex(1, 0), make_cuFloatComplex(-1, 0)
};

cuFloatComplex CX::gateMatrix[CX::matrixSize] = {
    make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0),
    make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0),
    make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0),
    make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0),
    
};

// Error checking macro
#define cudaCheckError() {                                           \
    cudaError_t e=cudaGetLastError();                                \
    if(e!=cudaSuccess) {                                             \
        std::cerr << "Cuda failure " << __FILE__ << ":" << __LINE__; \
        std::cerr << " '" << cudaGetErrorString(e) << "'\n";         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

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

// Kernel to normalize the statevector
__global__ void normalize(cuFloatComplex* statevector, int len) {
    __shared__ float norm;
    if (threadIdx.x == 0) norm = 0.0f;
    __syncthreads();

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < len) {
        atomicAdd(&norm, cuCabsf(statevector[idx]) * cuCabsf(statevector[idx]));
    }
    __syncthreads();

    if (threadIdx.x == 0) norm = sqrtf(norm);
    __syncthreads();

    if (idx < len) {
        statevector[idx] = make_cuFloatComplex(cuCrealf(statevector[idx]) / norm, cuCimagf(statevector[idx]) / norm);
    }
}



// Kernel to compute the probabilities
__global__ void compute_probabilities(float* probabilities, cuFloatComplex* statevector, int len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < len) {
        probabilities[idx] = cuCabsf(statevector[idx]) * cuCabsf(statevector[idx]);
    }
}

// Kernel to measure the statevector
__global__ void measure(int* result, float* probabilities, int len, unsigned long long seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);
    float random_number = curand_uniform(&state);

    float cumulative_probability = 0.0f;
    for (int i = 0; i < len; ++i) {
        cumulative_probability += probabilities[i];
        if (random_number < cumulative_probability) {
            *result = i;
            break;
        }
    }
}

int measure(cuFloatComplex *h_statevector, int num_qubits, int shot) {
    int len = 1 << num_qubits;

    //cuFloatComplex h_statevector[] = {
    //    make_cuFloatComplex(0.2, 0.0), make_cuFloatComplex(0.2, 0.0),
    //    make_cuFloatComplex(0.6, 0.0), make_cuFloatComplex(0.2, 0.0)
    //};

    // // Calculate the norm of the statevector
    // float norm = 0.0f;
    // for (int i = 0; i < len; ++i) {
    //     float real_part = cuCrealf(h_statevector[i]);
    //     float imag_part = cuCimagf(h_statevector[i]);
    //     norm += real_part * real_part + imag_part * imag_part;
    // }
    // norm = sqrtf(norm);

    // // Normalize the statevector
    // for (int i = 0; i < len; ++i) {
    //     float real_part = cuCrealf(h_statevector[i]);
    //     float imag_part = cuCimagf(h_statevector[i]);
    //     h_statevector[i] = make_cuFloatComplex(real_part / norm, imag_part / norm);
    // }

    // // Verify normalization
    // float sum = 0.0f;
    // for (int i = 0; i < len; ++i) {
    //     float mag = cuCabsf(h_statevector[i]);
    //     sum += mag * mag;
    // }

    // Device memory allocations
    cuFloatComplex* d_statevector;
    float* d_probabilities;
    int* d_result;
    // Allocate memory for random number generator states
    //curandState *d_states;
    
    cudaMalloc(&d_statevector, len * sizeof(cuFloatComplex));
    cudaMalloc(&d_probabilities, len * sizeof(float));
    cudaMalloc(&d_result, sizeof(int));
    //cudaMalloc((void **)&d_states, len * sizeof(curandState));

    // Copy statevector to device
    cudaMemcpy(d_statevector, h_statevector, len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Kernel launches
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;

    // Generate a random seed
    unsigned long long seed = time(NULL) + shot;

    normalize<<<gridSize, blockSize>>>(d_statevector, len);
    cudaCheckError();

    compute_probabilities<<<gridSize, blockSize>>>(d_probabilities, d_statevector, len);
    cudaCheckError();

    measure<<<1, 1>>>(d_result, d_probabilities, len, seed);
    cudaCheckError();

    // Copy result back to host
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Convert result to bitstring
    //std::string bitstring;
    //for (int i = num_qubits - 1; i >= 0; --i) {
    //    bitstring += ((h_result >> i) & 1) ? '1' : '0';
    //}

    // Clean up
    cudaFree(d_statevector);
    cudaFree(d_probabilities);
    cudaFree(d_result);

    //return binaryStringToInt(bitstring);
    return h_result;
}

// CUDA kernel to multiply a complex number against a matrix
__global__ void complexMatrixMultiply(const cuFloatComplex *matrix, const cuFloatComplex number, cuFloatComplex *result, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        
        // Perform complex multiplication
        float real = cuCrealf(matrix[index]) * cuCrealf(number) - cuCimagf(matrix[index]) * cuCimagf(number);
        float imag = cuCrealf(matrix[index]) * cuCimagf(number) + cuCimagf(matrix[index]) * cuCrealf(number);

        result[index] = make_cuFloatComplex(real, imag);
    }
}

int multiplication(cuFloatComplex *h_matrix, cuFloatComplex h_number, cuFloatComplex *h_result, int rows, int cols) {
    const int matrixSize = rows * cols;

    // Device (GPU) variables
    cuFloatComplex *d_matrix, *d_result;
    cuFloatComplex d_number = h_number;

    cudaMalloc((void**)&d_matrix, matrixSize * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_result, matrixSize * sizeof(cuFloatComplex));

    // Transfer matrix and number to device
    cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    complexMatrixMultiply<<<gridDim, blockDim>>>(d_matrix, d_number, d_result, rows, cols);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}

