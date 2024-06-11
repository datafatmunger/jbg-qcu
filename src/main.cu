
#include <assert.h>
#include <stdio.h>

#include "qsim.h"

int print_vector(cuFloatComplex *h_y, int rows) {
    // Print the result
    for (int i = 0; i < rows; ++i) {
        printf("%f+%fi ", cuCrealf(h_y[i]), cuCimagf(h_y[i]));
    }
    printf("\r\n");
    return 0;
}

int test_matVecMul(cuFloatComplex *h_y, int rows) {
    int cols = 4;

    // Hadamard matrix - JBG
    cuFloatComplex h_A[] = {
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0)
    };

    // Qubit zero state - JBG
    cuFloatComplex h_x[] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(0, 0)};
    

    // Matrix-vector multiplication
    matVecMul(h_A, h_x, h_y, rows, cols);

    assert(1.0 == cuCrealf(h_y[0]));
    assert(0.0 == cuCrealf(h_y[1]));
    assert(1.0 == cuCrealf(h_y[2]));
    assert(0.0 == cuCrealf(h_y[3]));

    return 0;
}

int main() {
    int rows = 4;

    // Allocate memory for 4 cuFloatComplex elements
    cuFloatComplex *h_y = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * rows);
    if (h_y == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    // Call the function with the allocated memory and size
    test_matVecMul(h_y, 4);
    print_vector(h_y, 4);
    
    // Free the allocated memory
    free(h_y);
    
    return 0;
}