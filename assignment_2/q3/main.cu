#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)


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
    // Shared memory for tiles
    // tileA: tileY x tileX, tileB: tileX x tileX
    extern __shared__ float shared_mem[];
    float *tileA = shared_mem;
    float *tileB = shared_mem + tileY * tileX;
    
    // Thread's position in the output matrix
    int row = blockIdx.y * tileY + threadIdx.y;
    int col = blockIdx.x * tileX + threadIdx.x;
    
    float sum = 0.0f;
    
    // Number of tiles needed to cover the K dimension (numACols = numBRows)
    int numTiles = (numACols + tileX - 1) / tileX;
    
    // Iterate over tiles along the K dimension
    for (int tile = 0; tile < numTiles; ++tile) {
        // Load tile of A into shared memory (tileY x tileX)
        int aRow = blockIdx.y * tileY + threadIdx.y;
        int aCol = tile * tileX + threadIdx.x;
        if (aRow < numARows && aCol < numACols) {
            tileA[threadIdx.y * tileX + threadIdx.x] = A[aRow * numACols + aCol];
        } else {
            tileA[threadIdx.y * tileX + threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory (tileX x tileX)
        // Threads with threadIdx.y < tileX participate in loading B
        if (threadIdx.y < tileX) {
            int bRow = tile * tileX + threadIdx.y;
            int bCol = blockIdx.x * tileX + threadIdx.x;
            if (bRow < numBRows && bCol < numBCols) {
                tileB[threadIdx.y * tileX + threadIdx.x] = B[bRow * numBCols + bCol];
            } else {
                tileB[threadIdx.y * tileX + threadIdx.x] = 0.0f;
            }
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < tileX && (tile * tileX + k) < numACols; ++k) {
            sum += tileA[threadIdx.y * tileX + k] * tileB[k * tileX + threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < numARows && col < numBCols) {
        C[row * numBCols + col] = sum;
    }
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
    float eps = 1e-3;  // Increased tolerance for floating-point precision differences
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

    size_t sizeA = numARows * numACols * sizeof(float);
    size_t sizeB = numBRows * numBCols * sizeof(float);
    size_t sizeC = numCRows * numCCols * sizeof(float);

    //@@ 1. Allocate in host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_ref = (float *)malloc(sizeC);

    //@@ 3. Initialize host memory
    for (int i = 0; i < numARows * numACols; i++)
        h_A[i] = 1.0f;
    for (int i = 0; i < numBRows * numBCols; i++)
        h_B[i] = 2.0f;

    //@@ 2. Allocate in device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, sizeA));
    CHECK(cudaMalloc((void **)&d_B, sizeB));
    CHECK(cudaMalloc((void **)&d_C, sizeC));

    //@@ 4. Copy from host memory to device memory
    CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Compute on CPU for reference
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    float cpu_time_ms;
    CHECK(cudaEventRecord(start));
    matrixMulCPU(h_A, h_B, h_ref, numARows, numACols, numBCols);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&cpu_time_ms, start, stop));

    // Print input matrix dimension
    printf("Input matrix dim: %dx%d x %dx%d\n", numARows, numACols, numBRows, numBCols);
    
    // Print CPU reference result (first few elements)
    printf("CPU reference result: ");
    int printCount = (numCRows * numCCols < 10) ? numCRows * numCCols : 10;
    for (int i = 0; i < printCount; i++) {
        printf("%.2f ", h_ref[i]);
    }
    if (numCRows * numCCols > 10) printf("...");
    printf("\n");
    printf("CPU timing: %.3f ms\n", cpu_time_ms);

    //@@ 5. Test basic GEMM kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numCRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CHECK(cudaEventRecord(start));
    gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gemm_time_ms;
    CHECK(cudaEventElapsedTime(&gemm_time_ms, start, stop));

    //@@ 7. Copy results from GPU to CPU
    CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify basic GEMM
    bool correct = verifyResults(h_ref, h_C, numCRows, numCCols);
    printf("CUDA gemm result: ");
    for (int i = 0; i < printCount; i++) {
        printf("%.2f ", h_C[i]);
    }
    if (numCRows * numCCols > 10) printf("...");
    printf("\n");
    printf("timing: %.3f ms\n", gemm_time_ms);
    if (!correct) {
        printf("WARNING: GEMM verification FAILED!\n");
    }

    // Test tiled GEMM with different tile sizes
    // Define tile sizes to test (tileX, tileY)
    int tileSizes[][2] = {{16, 16}, {32, 32}, {8, 8}};
    int numTileConfigs = 3;

    for (int t = 0; t < numTileConfigs; t++) {
        int tileX = tileSizes[t][0];
        int tileY = tileSizes[t][1];
        
        // Calculate shared memory size: tileA (tileY x tileX) + tileB (tileX x tileX)
        size_t sharedMemSize = (tileY * tileX + tileX * tileX) * sizeof(float);
        
        // Grid dimensions for tiled kernel
        dim3 tiledBlocksPerGrid((numCCols + tileX - 1) / tileX,
                                (numCRows + tileY - 1) / tileY);
        dim3 tiledThreadsPerBlock(tileX, tileY);
        
        CHECK(cudaEventRecord(start));
        tiled_gemm<<<tiledBlocksPerGrid, tiledThreadsPerBlock, sharedMemSize>>>(
            d_A, d_B, d_C, numARows, numACols, numBRows, numBCols, tileX, tileY);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float tiled_time_ms;
        CHECK(cudaEventElapsedTime(&tiled_time_ms, start, stop));
        
        // Copy results
        CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        
        // Verify tiled GEMM
        bool tiled_correct = verifyResults(h_ref, h_C, numCRows, numCCols);
        printf("CUDA tiled_gemm with tile [%d, %d] result: ", tileX, tileY);
        for (int i = 0; i < printCount; i++) {
            printf("%.2f ", h_C[i]);
        }
        if (numCRows * numCCols > 10) printf("...");
        printf("\n");
        printf("timing: %.3f ms\n", tiled_time_ms);
        if (!tiled_correct) {
            printf("WARNING: Tiled GEMM [%d, %d] verification FAILED!\n", tileX, tileY);
        }
    }

    //@@ 9. Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    //@@ 10. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
