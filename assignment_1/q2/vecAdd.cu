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

//@@ CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}


//@@ CUDA kernel for vector multiplication
__global__ void matrixMul(const float *A, const float *B, float *C, int numARows, int numACols, int numBCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBCols) {
        float sum = 0.0f;
        for (int i = 0; i < numACols; ++i) {
            sum += A[row * numACols + i] * B[i * numBCols + col];
        }
        C[row * numBCols + col] = sum;
    }
}

//@@ CPU version of vector addition
void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}


//@@ CPU version of vector multiplication
void matrixMulCPU(const float *A, const float *B, float *C,
                  int numARows, int numACols, int numBCols) {
    for (int row = 0; row < numARows; ++row) {
        for (int col = 0; col < numBCols; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < numACols; ++i) {
                sum += A[row * numACols + i] * B[i * numBCols + col];
            }
            C[row * numBCols + col] = sum;
        }
    }
}

//@@ Comparision of CPU and GPU implementation
bool verifyResults(const float *ref, const float *gpu, int numCRows, int numCCols) {
    float eps = 1e-5;
    for (int i = 0; i < numCRows * numCCols; ++i) {
        if (fabs(ref[i] - gpu[i]) > eps) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int n = 512;

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    size_t size = n * sizeof(float);

    //@@ 1. Allocate in host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

    //@@ 3. Initialize host memory
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    //@@ 2. Allocate in device memory.
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));

    //@@ 4. Copy from host memory to device memory
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    //@@ 5. Initialize thread block and thread grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    //@@ 6. Invoke the CUDA kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    //@@ 7. Copy results from GPU to CPU
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    
    vectorAddCPU(h_A, h_B, h_ref, n);

    //@@ 8. Compare the results with the CPU reference result
    bool correct = verifyResults(h_ref, h_C, n);
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
