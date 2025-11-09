#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)


//@@ CUDA kernel for vector multiplication
__global__ void matrixMul(const double *A, const double *B, double *C, int numARows, int numACols, int numBCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBCols) {
        double sum = 0.0;
        for (int i = 0; i < numACols; ++i) {
            sum += A[row * numACols + i] * B[i * numBCols + col];
        }
        C[row * numBCols + col] = sum;
        // printf("[+] C[%d] = %f\n", (row * numBCols + col), sum);
    }
}


//@@ CPU version of vector multiplication
void matrixMulCPU(const double *A, const double *B, double *C,
                  int numARows, int numACols, int numBCols) {
    for (int row = 0; row < numARows; ++row) {
        for (int col = 0; col < numBCols; ++col) {
            double sum = 0.0;
            for (int i = 0; i < numACols; ++i) {
                sum += A[row * numACols + i] * B[i * numBCols + col];
            }
            C[row * numBCols + col] = sum;
        }
    }
}

//@@ Comparision of CPU and GPU implementation
bool verifyResults(const double *ref, const double *gpu, int numCRows, int numCCols) {
    double eps = 1e-5;
    for (int i = 0; i < numCRows * numCCols; ++i) {
        if (fabs(ref[i] - gpu[i]) > eps) {
            printf("Mismatch at index %d: CPU = %lf, GPU = %lf\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int numARows = 4, numACols = 4;
    int numBRows = 4, numBCols = 4;

    // Allow user to override dimensions
    if (argc == 5) {
        numARows = atoi(argv[1]);
        numACols = atoi(argv[2]);
        numBRows = atoi(argv[3]);
        numBCols = atoi(argv[4]);
    }

    // Validate matrix dimensions for multiplication
    if (numACols != numBRows) {
        printf("Error: Number of columns of A must equal number of rows of B.\n");
        return -1;
    }

    int numCRows = numARows;
    int numCCols = numBCols;

    size_t sizeA = numARows * numACols * sizeof(double);
    size_t sizeB = numBRows * numBCols * sizeof(double);
    size_t sizeC = numCRows * numCCols * sizeof(double);

    //@@ 1. Allocate in host memory
    double *h_A = (double *)malloc(sizeA);
    double *h_B = (double *)malloc(sizeB);
    double *h_C = (double *)malloc(sizeC);
    double *h_ref = (double *)malloc(sizeC);

    //@@ 3. Initialize host memory
    for (int i = 0; i < numARows * numACols; i++)
        h_A[i] = 1.0;
    for (int i = 0; i < numBRows * numBCols; i++)
        h_B[i] = 2.0;

    //@@ 2. Allocate in device memory
    double *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, sizeA));
    CHECK(cudaMalloc((void **)&d_B, sizeB));
    CHECK(cudaMalloc((void **)&d_C, sizeC));

    //@@ 4. Copy from host memory to device memory
    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    //@@ 5. Initialize thread block and thread grid
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numCRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //@@ 6. Invoke the CUDA kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                  numARows, numACols, numBCols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    //@@ 7. Copy results from GPU to CPU
    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute on CPU for reference
    matrixMulCPU(h_A, h_B, h_ref, numARows, numACols, numBCols);

    //@@ 8. Compare the results with the CPU reference result
    bool correct = verifyResults(h_ref, h_C, numCRows, numCCols);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    //@@ 9. Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    //@@ 10. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
