#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA fragment dimensions (16x16x16 for half precision)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)


//@@ Kernel to convert float to half precision
__global__ void convertFp32ToFp16(half *out, const float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

//@@ CUDA kernel for basic matrix multiplication (GEMM)
__global__ void gemm(const float *A, const float *B, float *C, int numARows, int numACols, int numBCols) {
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

//@@ CUDA kernel for tiled matrix multiplication (Tiled GEMM)
__global__ void tiled_gemm(const float *A, const float *B, float *C, 
                           int numARows, int numACols, int numBRows, int numBCols,
                           int tileX, int tileY) {
    extern __shared__ float shared_mem[];
    float *tileA = shared_mem;
    float *tileB = shared_mem + tileY * tileX;
    
    int row = blockIdx.y * tileY + threadIdx.y;
    int col = blockIdx.x * tileX + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (numACols + tileX - 1) / tileX;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        int aRow = blockIdx.y * tileY + threadIdx.y;
        int aCol = tile * tileX + threadIdx.x;
        if (aRow < numARows && aCol < numACols) {
            tileA[threadIdx.y * tileX + threadIdx.x] = A[aRow * numACols + aCol];
        } else {
            tileA[threadIdx.y * tileX + threadIdx.x] = 0.0f;
        }
        
        if (threadIdx.y < tileX) {
            int bRow = tile * tileX + threadIdx.y;
            int bCol = blockIdx.x * tileX + threadIdx.x;
            if (bRow < numBRows && bCol < numBCols) {
                tileB[threadIdx.y * tileX + threadIdx.x] = B[bRow * numBCols + bCol];
            } else {
                tileB[threadIdx.y * tileX + threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        for (int k = 0; k < tileX && (tile * tileX + k) < numACols; ++k) {
            sum += tileA[threadIdx.y * tileX + k] * tileB[k * tileX + threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < numARows && col < numBCols) {
        C[row * numBCols + col] = sum;
    }
}

//@@ CUDA kernel for WMMA Tensor Core matrix multiplication
// Uses half precision inputs and float accumulator (mixed precision)
// Each warp computes one 16x16 output tile
__global__ void wmma_gemm(const half *A, const half *B, float *C,
                          int numARows, int numACols, int numBCols) {
    // Each block (one warp) handles a 16x16 output tile
    int tileRow = blockIdx.y * WMMA_M;
    int tileCol = blockIdx.x * WMMA_N;
    
    if (tileRow >= numARows || tileCol >= numBCols) return;
    
    // Declare fragments for matrix A, B, and accumulator C
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < numACols; k += WMMA_K) {
        // Load matrix A and B fragments
        wmma::load_matrix_sync(a_frag, A + tileRow * numACols + k, numACols);
        wmma::load_matrix_sync(b_frag, B + k * numBCols + tileCol, numBCols);
        
        // Perform MMA: C = A * B + C
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result to global memory
    wmma::store_matrix_sync(C + tileRow * numBCols + tileCol, c_frag, numBCols, wmma::mem_row_major);
}


//@@ CPU version of matrix multiplication
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

//@@ Comparison of CPU and GPU implementation
bool verifyResults(const float *ref, const float *gpu, int numCRows, int numCCols, float eps = 1e-3) {
    for (int i = 0; i < numCRows * numCCols; ++i) {
        float diff = fabs(ref[i] - gpu[i]);
        float relError = (fabs(ref[i]) > 1e-6) ? diff / fabs(ref[i]) : diff;
        if (diff > eps && relError > 1e-4) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f, diff = %f\n", i, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

//@@ Compute accuracy metrics for WMMA (uses lower precision)
void computeAccuracyMetrics(const float *ref, const float *result, int size,
                            float *maxError, float *avgError) {
    float sumError = 0.0f;
    *maxError = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        float diff = fabs(ref[i] - result[i]);
        if (diff > *maxError) *maxError = diff;
        sumError += diff;
    }
    *avgError = sumError / size;
}

int main(int argc, char **argv) {
    int numARows = 1024, numACols = 2048;
    int numBRows = 2048, numBCols = 1024;

    // Allow user to override dimensions
    if (argc == 5) {
        numARows = atoi(argv[1]);
        numACols = atoi(argv[2]);
        numBRows = atoi(argv[3]);
        numBCols = atoi(argv[4]);
    }

    // Validate matrix dimensions
    if (numACols != numBRows) {
        printf("Error: Number of columns of A must equal number of rows of B.\n");
        return -1;
    }

    int numCRows = numARows;
    int numCCols = numBCols;

    size_t sizeA = numARows * numACols * sizeof(float);
    size_t sizeB = numBRows * numBCols * sizeof(float);
    size_t sizeC = numCRows * numCCols * sizeof(float);
    size_t sizeA_half = numARows * numACols * sizeof(half);
    size_t sizeB_half = numBRows * numBCols * sizeof(half);

    //@@ 1. Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_ref = (float *)malloc(sizeC);

    //@@ 2. Initialize host memory
    srand(42);
    for (int i = 0; i < numARows * numACols; i++)
        h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < numBRows * numBCols; i++)
        h_B[i] = (float)(rand() % 100) / 100.0f;

    //@@ 3. Allocate device memory
    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half;
    CHECK(cudaMalloc((void **)&d_A, sizeA));
    CHECK(cudaMalloc((void **)&d_B, sizeB));
    CHECK(cudaMalloc((void **)&d_C, sizeC));
    CHECK(cudaMalloc((void **)&d_A_half, sizeA_half));
    CHECK(cudaMalloc((void **)&d_B_half, sizeB_half));

    //@@ 4. Copy from host to device
    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    //@@ 5. Convert float to half precision for WMMA
    int convBlock = 256;
    int convGridA = (numARows * numACols + convBlock - 1) / convBlock;
    int convGridB = (numBRows * numBCols + convBlock - 1) / convBlock;
    convertFp32ToFp16<<<convGridA, convBlock>>>(d_A_half, d_A, numARows * numACols);
    convertFp32ToFp16<<<convGridB, convBlock>>>(d_B_half, d_B, numBRows * numBCols);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    printf("Matrix dim: A(%dx%d) x B(%dx%d) = C(%dx%d)\n", 
           numARows, numACols, numBRows, numBCols, numCRows, numCCols);

    //@@ 6. CPU reference
    float cpu_time_ms;
    CHECK(cudaEventRecord(start));
    matrixMulCPU(h_A, h_B, h_ref, numARows, numACols, numBCols);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&cpu_time_ms, start, stop));
    printf("CPU: %.3f ms\n", cpu_time_ms);

    //@@ 7. Test basic GEMM kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCCols + 15) / 16, (numCRows + 15) / 16);

    CHECK(cudaEventRecord(start));
    gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gemm_time_ms;
    CHECK(cudaEventElapsedTime(&gemm_time_ms, start, stop));

    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    bool correct = verifyResults(h_ref, h_C, numCRows, numCCols);
    printf("GEMM: %.3f ms %s\n", gemm_time_ms, correct ? "" : "[FAILED]");

    //@@ 8. Test tiled GEMM kernel
    int tileX = 16, tileY = 16;
    size_t sharedMemSize = (tileY * tileX + tileX * tileX) * sizeof(float);
    dim3 tiledGrid((numCCols + tileX - 1) / tileX, (numCRows + tileY - 1) / tileY);
    dim3 tiledBlock(tileX, tileY);

    CHECK(cudaEventRecord(start));
    tiled_gemm<<<tiledGrid, tiledBlock, sharedMemSize>>>(
        d_A, d_B, d_C, numARows, numACols, numBRows, numBCols, tileX, tileY);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float tiled_time_ms;
    CHECK(cudaEventElapsedTime(&tiled_time_ms, start, stop));

    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    correct = verifyResults(h_ref, h_C, numCRows, numCCols);
    printf("Tiled GEMM: %.3f ms %s\n", tiled_time_ms, correct ? "" : "[FAILED]");

    //@@ 9. Test WMMA Tensor Core kernel
    // Grid: one warp (32 threads) per 16x16 output tile
    dim3 wmmaBlock(32, 1);
    dim3 wmmaGrid((numCCols + WMMA_N - 1) / WMMA_N, (numCRows + WMMA_M - 1) / WMMA_M);

    CHECK(cudaEventRecord(start));
    wmma_gemm<<<wmmaGrid, wmmaBlock>>>(d_A_half, d_B_half, d_C, numARows, numACols, numBCols);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float wmma_time_ms;
    CHECK(cudaEventElapsedTime(&wmma_time_ms, start, stop));

    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // WMMA uses half precision, so use larger tolerance
    float maxErr, avgErr;
    computeAccuracyMetrics(h_ref, h_C, numCRows * numCCols, &maxErr, &avgErr);
    printf("WMMA: %.3f ms (maxErr=%.6f, avgErr=%.6f)\n", wmma_time_ms, maxErr, avgErr);

    //@@ 10. Print fragment info for assignment question
    int numFragM = (numARows + WMMA_M - 1) / WMMA_M;
    int numFragN = (numBCols + WMMA_N - 1) / WMMA_N;
    int numFragK = (numACols + WMMA_K - 1) / WMMA_K;
    printf("\nFragment dimensions: %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Fragments: M=%d, N=%d, K=%d\n", numFragM, numFragN, numFragK);
    printf("Total fragments: A=%d, B=%d, C=%d\n", 
           numFragM * numFragK, numFragK * numFragN, numFragM * numFragN);

    //@@ 11. Summary
    printf("\nSummary:\n");
    printf("CPU:    %.3f ms\n", cpu_time_ms);
    printf("GEMM:   %.3f ms (%.2fx)\n", gemm_time_ms, cpu_time_ms / gemm_time_ms);
    printf("Tiled:  %.3f ms (%.2fx)\n", tiled_time_ms, cpu_time_ms / tiled_time_ms);
    printf("WMMA:   %.3f ms (%.2fx)\n", wmma_time_ms, cpu_time_ms / wmma_time_ms);

    //@@ 12. Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
