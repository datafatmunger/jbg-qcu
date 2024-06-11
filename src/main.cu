
#include <stdio.h>
#include "qsim.h"

int test_matVecMul() {
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
    //for (int i = 0; i < rows; ++i) {
    //    printf("%f+%fi ", cuCrealf(h_y[i]), cuCimagf(h_y[i]));
    //}
    printf("\r\n");

    return 0;
}


int main() {
    test_matVecMul();
    return 0;
}