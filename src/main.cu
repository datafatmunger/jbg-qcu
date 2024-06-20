
#include <assert.h>
#include <stdio.h>

#include "qsim.h"

int print_vector(cuFloatComplex *h_y, int rows) {
    // Print the result
    for (int i = 0; i < rows; ++i) {
        std::cout << cuCrealf(h_y[i]) << "+" << cuCimagf(h_y[i]) << "i ";
    }
    std::cout << "\n";
    return 0;
}

int print_matrix(cuFloatComplex *h_C, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) { 
            std::cout << cuCrealf(h_C[i * cols + j]) << "+" << cuCimagf(h_C[i * cols + j]) << "i ";
        }
        std::cout << "\n";
    }
    return 0;
}

// Function to compare two floating-point numbers with a given epsilon
bool isAlmostEqual(float a, float b, float epsilon) {
    return std::fabs(a - b) <= epsilon;
}

int test_matVecMul() {
    int rows = 4;

    // Allocate memory for 4 cuFloatComplex elements
    cuFloatComplex *h_y = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * rows);
    if (h_y == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

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

    // Free the allocated memory
    free(h_y);

    return 0;
}

int test_cx() {
    int rows = 4;

    // Allocate memory for 4 cuFloatComplex elements
    cuFloatComplex *h_y = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * rows);
    if (h_y == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    int cols = 4;

    // Qubit zero state - JBG
    cuFloatComplex h_x[] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0), make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0)};
    

    // Matrix-vector multiplication
    matVecMul(CX::gateMatrix, h_x, h_y, rows, cols);

    assert(1.0 == cuCrealf(h_y[0]));
    assert(0.0 == cuCrealf(h_y[1]));
    assert(0.0 == cuCrealf(h_y[2]));
    assert(1.0 == cuCrealf(h_y[3]));

    // Free the allocated memory
    free(h_y);

    return 0;
}

int test_measure() {
     int num_qubits = 2;
     cuFloatComplex h_statevector[] = {
        make_cuFloatComplex(0.2, 0.0), make_cuFloatComplex(0.2, 0.0),
        make_cuFloatComplex(0.6, 0.0), make_cuFloatComplex(0.2, 0.0)
    };

    int counts[] = {0, 0, 0, 0};
    int shots = 1000;
    for(int shot = 0; shot < shots; shot++) {
        int result = measure(h_statevector, num_qubits, shot);
        counts[result] += 1;
    }

    float epsilon = 0.1f;
    assert(isAlmostEqual(float(counts[0]) / shots, .083, epsilon));
    assert(isAlmostEqual(float(counts[1]) / shots, .083, epsilon));
    assert(isAlmostEqual(float(counts[2]) / shots, .75, epsilon));
    assert(isAlmostEqual(float(counts[3]) / shots, .083, epsilon));

    return 0;
}

int test_tensor() {
    const int aRows = 2;
    const int aCols = 2;
    const int bRows = 2;
    const int bCols = 2;

    cuFloatComplex *h_A = H::gateMatrix;
    cuFloatComplex *h_B = I::gateMatrix;
    cuFloatComplex h_C[aRows * bRows * aCols * bCols];

    tensorProduct(h_A, h_B, h_C, aRows, aCols, bRows, bCols);

    // Expected result
    cuFloatComplex h_R[] = {
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),
        make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0),  make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),  make_cuFloatComplex(1, 0),  make_cuFloatComplex(0, 0), make_cuFloatComplex(-1, 0)
    };

    //print_matrix(h_C, aRows * bRows, aCols * bCols);

    for (int i = 0; i < aRows * bRows; ++i) {
        for (int j = 0; j < aCols * bCols; ++j) {
            assert(cuCrealf(h_C[i * aCols * bCols + j]) == cuCrealf(h_R[i * aCols * bCols + j]));
            assert(cuCimagf(h_C[i * aCols * bCols + j]) == cuCimagf(h_R[i * aCols * bCols + j]));
        }

    }

    return 0;
}

int test_superpos() {
    cuFloatComplex *h_y = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * 2);
    cuFloatComplex h_x[2] = {make_cuFloatComplex(1, 0), make_cuFloatComplex(0, 0)};

    // Apply H Gate
    matVecMul(H::gateMatrix, h_x, h_y, 2, 2);
    cuFloatComplex *h_result =(cuFloatComplex *)malloc(sizeof(cuFloatComplex) * 2);
    cuFloatComplex h_number = make_cuFloatComplex(1 / sqrtf(2), 0);
    multiplication(h_y, h_number, h_result, 2, 1);

    std::cout << "Statevector --" << std::endl;
    print_vector(h_result, 2);
    std::cout << std::endl;

    int num_qubits = 1;
    int counts[] = {0, 0, 0, 0};
    int shots = 1000;
    for(int shot = 0; shot < shots; shot++) {
        int result = measure(h_result, num_qubits, shot);
        counts[result] += 1;
    }

    float epsilon = 0.1f;
    assert(isAlmostEqual(float(counts[0]) / shots, .5, epsilon));
    assert(isAlmostEqual(float(counts[1]) / shots, .5, epsilon));

    // Free the allocated memory
    free(h_y);
    free(h_result);

    return 0;
}

int test_bell() {
    const int num_qubits = 2;

    const int aRows = num_qubits;
    const int aCols = num_qubits;
    const int bRows = num_qubits;
    const int bCols = num_qubits;

    // Initial statevector
    cuFloatComplex v_0[2 * num_qubits] = {
        make_cuFloatComplex(1, 0),
        make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),
        make_cuFloatComplex(0, 0),
    };

    // Apply H gate
    cuFloatComplex *m_0 =(cuFloatComplex *)malloc(sizeof(cuFloatComplex) * aRows * bRows * aCols * bCols);
    tensorProduct(H::gateMatrix, I::gateMatrix, m_0, aRows, aCols, bRows, bCols);

    print_matrix(m_0, aRows * bRows, aCols * bCols);
    std::cout << std::endl;

    cuFloatComplex *m_1 =(cuFloatComplex *)malloc(sizeof(cuFloatComplex) * aRows * bRows * aCols * bCols);
    cuFloatComplex h_number = make_cuFloatComplex(1 / sqrtf(2), 0);
    multiplication(m_0, h_number, m_1, aRows * bRows, aCols * bCols);

    print_matrix(m_1, aRows * bRows, aCols * bCols);
    std::cout << std::endl;

    cuFloatComplex *v_1 = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * 2 * num_qubits);
    matVecMul(m_1, v_0, v_1, aRows * bRows, aCols * bCols);

    std::cout << "Statevector --" << std::endl;
    print_vector(v_1, 2 * num_qubits);
    std::cout << std::endl;

    // Apply CX gate
    cuFloatComplex *v_2 = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * 2 * num_qubits);
    matVecMul(CX::gateMatrix, v_1, v_2, aRows * bRows, aCols * bCols);

    std::cout << "Statevector --" << std::endl;
    print_vector(v_2, 2 * num_qubits);
    std::cout << std::endl;

    // Measure
    int counts[] = {0, 0, 0, 0};
    int shots = 1000;
    for(int shot = 0; shot < shots; shot++) {
        int result = measure(v_2, num_qubits, shot);
        counts[result] += 1;
    }

    std::cout << "00: " << counts[0] << " " << 1.0f * counts[0] / shots <<std::endl;
    std::cout << "01: " << counts[1] << " " << 1.0f * counts[1] / shots <<std::endl;
    std::cout << "10: " << counts[2] << " " << 1.0f * counts[2] / shots <<std::endl;
    std::cout << "11: " << counts[3] << " " << 1.0f * counts[3] / shots <<std::endl;

    // Test
    float epsilon = 0.1f;
    assert(isAlmostEqual(float(counts[0]) / shots, .5, epsilon));
    assert(isAlmostEqual(float(counts[1]) / shots, 0, epsilon));
    assert(isAlmostEqual(float(counts[2]) / shots, 0, epsilon));
    assert(isAlmostEqual(float(counts[3]) / shots, .5, epsilon));

    // Cleanup
    free(m_0);
    free(m_1);
    free(v_1);
    free(v_2);

    return 0;
}

int main() {

    test_matVecMul();
    test_tensor();
    test_measure();
    test_superpos();
    test_cx();
    test_bell();

    return 0;
}