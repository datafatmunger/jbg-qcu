#ifndef QSIM_H
#define QSIM_H

#include <cuda_runtime.h>
#include <cuComplex.h>

void matVecMul(cuFloatComplex *h_A, cuFloatComplex *h_x, cuFloatComplex *h_y, int rows, int cols);

#endif // QSIM_H