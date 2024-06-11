#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>

// Error checking macro
#define cudaCheckError() {                                           \
    cudaError_t e=cudaGetLastError();                                \
    if(e!=cudaSuccess) {                                             \
        std::cerr << "Cuda failure " << __FILE__ << ":" << __LINE__; \
        std::cerr << " '" << cudaGetErrorString(e) << "'\n";         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

__global__ void setup_kernel(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void generate_random_numbers(curandState *state, float *randomNumbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Generate a random number using the curand_uniform function
    float randNum = curand_uniform(&state[idx]);
    
    // Store the random number in the output array
    randomNumbers[idx] = randNum;
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
__global__ void measure(float *randomNumbers, int* result, float* probabilities, int len, unsigned long long seed) {
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //curandState state;
    //curand_init(seed + idx, 0, 0, &state);
    //float random_number = curand_uniform(&state);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // // Generate a random number using the curand_uniform function
    float random_number = randomNumbers[idx];

    float cumulative_probability = 0.0f;
    for (int i = 0; i < len; ++i) {
        cumulative_probability += probabilities[i];
        if (random_number < cumulative_probability) {
            *result = i;
            break;
        }
    }
}

void checkNormalization(cuFloatComplex* statevector, int len) {
    float norm = 0.0;
    for (int i = 0; i < len; ++i) {
        norm += cuCabsf(statevector[i]) * cuCabsf(statevector[i]);
    }
    if (fabs(norm - 1.0) > 1e-6) {
        std::cerr << "Statevector must be normalized\n";
        exit(EXIT_FAILURE);
    }
}

int binaryStringToInt(const std::string& binaryStr) {
    int result = 0;
    for (char bit : binaryStr) {
        result <<= 1; // shift left by 1 bit
        result += (bit - '0'); // add the current bit
    }
    return result;
}

int test_measure() {
    int num_qubits = 2;
    int len = 1 << num_qubits;

    cuFloatComplex h_statevector[] = {
        make_cuFloatComplex(0.2, 0.0), make_cuFloatComplex(0.2, 0.0),
        make_cuFloatComplex(0.6, 0.0), make_cuFloatComplex(0.2, 0.0)
    };

    // Calculate the norm of the statevector
    float norm = 0.0f;
    for (int i = 0; i < len; ++i) {
        float real_part = cuCrealf(h_statevector[i]);
        float imag_part = cuCimagf(h_statevector[i]);
        norm += real_part * real_part + imag_part * imag_part;
    }
    norm = sqrtf(norm);

    // Normalize the statevector
    for (int i = 0; i < len; ++i) {
        float real_part = cuCrealf(h_statevector[i]);
        float imag_part = cuCimagf(h_statevector[i]);
        h_statevector[i] = make_cuFloatComplex(real_part / norm, imag_part / norm);
    }

    // Verify normalization
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float mag = cuCabsf(h_statevector[i]);
        sum += mag * mag;
    }

    // Device memory allocations
    cuFloatComplex* d_statevector;
    float* d_probabilities;
    int* d_result;
    // Allocate memory for random number generator states
    curandState *d_states;
    
    cudaMalloc(&d_statevector, len * sizeof(cuFloatComplex));
    cudaMalloc(&d_probabilities, len * sizeof(float));
    cudaMalloc(&d_result, sizeof(int));
    cudaMalloc((void **)&d_states, len * sizeof(curandState));

    // Copy statevector to device
    cudaMemcpy(d_statevector, h_statevector, len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Kernel launches
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;

    // Generate a random seed
    unsigned long long seed = time(NULL);

    // Setup the kernel with random states
    setup_kernel<<<gridSize, blockSize>>>(d_states, seed);
    cudaCheckError();

    // Allocate memory for the random numbers generated
    float *randomNumbers;
    cudaMalloc((void **)&randomNumbers, len * sizeof(float));

    // Generate random numbers using the initialized states
    generate_random_numbers<<<gridSize, blockSize>>>(d_states, randomNumbers);
    cudaCheckError();

    normalize<<<gridSize, blockSize>>>(d_statevector, len);
    cudaCheckError();

    compute_probabilities<<<gridSize, blockSize>>>(d_probabilities, d_statevector, len);
    cudaCheckError();

    measure<<<1, 1>>>(randomNumbers, d_result, d_probabilities, len, time(NULL));
    cudaCheckError();

    // Copy the generated random numbers back to the host if needed
    float *hostRandomNumbers = (float *)malloc(len * sizeof(float));
    cudaMemcpy(hostRandomNumbers, randomNumbers, len * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < len; i++) {
        std::cout << i << ": " << hostRandomNumbers[i] << std::endl;
    }

    // Copy result back to host
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Convert result to bitstring
    std::string bitstring;
    for (int i = num_qubits - 1; i >= 0; --i) {
        bitstring += ((h_result >> i) & 1) ? '1' : '0';
    }

    // Clean up
    cudaFree(d_statevector);
    cudaFree(d_probabilities);
    cudaFree(d_result);
    cudaFree(d_states);
    cudaFree(randomNumbers);

    free(hostRandomNumbers);

    return binaryStringToInt(bitstring);
}


int main() {
    int counts[] = {0, 0, 0, 0};
    int shots = 1000;
    for(int shot = 0; shot < shots; shot++) {
        int result = test_measure();
        counts[result] += 1;
    }
    std::cout << "00: " << counts[0] << " " << 1.0f * counts[0] / shots <<std::endl;
    std::cout << "01: " << counts[1] << " " << 1.0f * counts[1] / shots <<std::endl;
    std::cout << "10: " << counts[2] << " " << 1.0f * counts[2] / shots <<std::endl;
    std::cout << "11: " << counts[3] << " " << 1.0f * counts[3] / shots <<std::endl;
}
