#ifndef QSIM_H
#define QSIM_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

#include <cmath>
#include <iostream>
#include <time.h>

class I {
public:
    static const int matrixSize = 4;  // Define the size of the array
    static cuFloatComplex gateMatrix[matrixSize];   // Declare the static array
};

class H {
public:
    static const int matrixSize = 4;  // Define the size of the array
    static cuFloatComplex gateMatrix[matrixSize];   // Declare the static array
};

class CX {
public:
    static const int matrixSize = 16;  // Define the size of the array
    static cuFloatComplex gateMatrix[matrixSize];   // Declare the static array
};

void matVecMul(cuFloatComplex *h_A, cuFloatComplex *h_x, cuFloatComplex *h_y, int rows, int cols);
void tensorProduct(cuFloatComplex* h_A, cuFloatComplex* h_B, cuFloatComplex* h_C, int aRows, int aCols, int bRows, int bCols);
int measure(cuFloatComplex *h_statevector, int num_qubits, int shot);
int multiplication(cuFloatComplex *h_matrix, cuFloatComplex h_number, cuFloatComplex *h_result, int rows, int cols);

#endif // QSIM_H