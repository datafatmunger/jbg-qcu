#include <curand_kernel.h>
#include <iostream>

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

// Function to swap two elements
void swap(float *xp, float *yp) {
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

// Function to perform Bubble Sort
void bubbleSort(float arr[], int n) {
    int i, j;
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
            }
        }
    }
}

// Function to count unique numbers in a sorted array
int countUnique(float arr[], int n) {
    if (n == 0) return 0;
    
    int count = 1; // there's at least one unique element
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[i-1]) {
            count++;
        }
    }
    return count;
}

int main()
{
    // Number of threads per block
    int threadsPerBlock = 256;
    // Number of blocks in the grid
    int blocksPerGrid = 256;
    // Total number of threads
    int numThreads = threadsPerBlock * blocksPerGrid;

    // Allocate memory for random number generator states
    curandState *devStates;
    cudaMalloc((void **)&devStates, numThreads * sizeof(curandState));

    // Generate a random seed
    unsigned long long seed = time(NULL);

    // Setup the kernel with random states
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(devStates, seed);
    
    // Allocate memory for the random numbers generated
    float *randomNumbers;
    cudaMalloc((void **)&randomNumbers, numThreads * sizeof(float));

    // Generate random numbers using the initialized states
    generate_random_numbers<<<blocksPerGrid, threadsPerBlock>>>(devStates, randomNumbers);
    
    // Synchronize to ensure all kernel executions are complete
    cudaDeviceSynchronize();

    // Copy the generated random numbers back to the host if needed
    float *hostRandomNumbers = (float *)malloc(numThreads * sizeof(float));
    cudaMemcpy(hostRandomNumbers, randomNumbers, numThreads * sizeof(float), cudaMemcpyDeviceToHost);

    //for(int i = 0; i < numThreads; i++) {
    //    std::cout << i << ": " << hostRandomNumbers[i] << std::endl;
    //}

    bubbleSort(hostRandomNumbers, numThreads);

    //for(int i = 0; i < numThreads; i++) {
    //    std::cout << i << ": " << hostRandomNumbers[i] << std::endl;
    //}

    int uniqueCount = countUnique(hostRandomNumbers, numThreads);
    std::cout << "Number of unique elements: " << uniqueCount << " in " << numThreads << std::endl;

    // Cleanup
    cudaFree(devStates);
    cudaFree(randomNumbers);

    free(hostRandomNumbers);

    return 0;
}
