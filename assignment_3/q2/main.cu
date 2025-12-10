#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

//@@ CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

//@@ CPU version of vector addition
void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

//@@ Comparison of CPU and GPU implementation
bool verifyResults(const float *ref, const float *gpu, int n) {
    float eps = 1e-5;
    for (int i = 0; i < n; ++i) {
        if (fabs(ref[i] - gpu[i]) > eps) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

//@@ Non-streamed version (baseline)
float runNonStreamed(float *h_A, float *h_B, float *h_C, int n) {
    size_t size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    CHECK(cudaEventRecord(start));
    
    // Synchronous copies
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CHECK(cudaGetLastError());
    
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    
    return milliseconds;
}

//@@ Streamed version
float runStreamed(float *h_A, float *h_B, float *h_C, int n, int S_seg, int NUM_STREAMS) {
    size_t size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));
    
    cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    int threadsPerBlock = 256;
    
    CHECK(cudaEventRecord(start));
    
    int numSegments = (n + S_seg - 1) / S_seg;
    
    for (int i = 0; i < numSegments; i++) {
        int streamId = i % NUM_STREAMS;
        int offset = i * S_seg;
        int segmentSize = (offset + S_seg <= n) ? S_seg : (n - offset);
        size_t segmentBytes = segmentSize * sizeof(float);
        int blocksPerGrid = (segmentSize + threadsPerBlock - 1) / threadsPerBlock;
        
        CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, segmentBytes, 
                              cudaMemcpyHostToDevice, streams[streamId]));
        CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, segmentBytes, 
                              cudaMemcpyHostToDevice, streams[streamId]));
        
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamId]>>>
                  (d_A + offset, d_B + offset, d_C + offset, segmentSize);
        
        CHECK(cudaMemcpyAsync(h_C + offset, d_C + offset, segmentBytes, 
                              cudaMemcpyDeviceToHost, streams[streamId]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    
    return milliseconds;
}

int main(int argc, char **argv) {
    // Default parameters
    int n = 1 << 24;  // 16M elements
    int S_seg = 1 << 20;  // 1M segment size
    int NUM_STREAMS = 4;
    bool benchmark_mode = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            S_seg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            NUM_STREAMS = atoi(argv[++i]);
        }
    }
    
    size_t size = n * sizeof(float);
    
    // Allocate pinned memory
    float *h_A, *h_B, *h_C, *h_ref;
    CHECK(cudaMallocHost((void **)&h_A, size));
    CHECK(cudaMallocHost((void **)&h_B, size));
    CHECK(cudaMallocHost((void **)&h_C, size));
    h_ref = (float *)malloc(size);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    if (benchmark_mode) {
        // Warmup run
        runNonStreamed(h_A, h_B, h_C, n);
        runStreamed(h_A, h_B, h_C, n, S_seg, NUM_STREAMS);

        printf("### VECTOR_SIZE_BENCHMARK ###\n");
        printf("vector_size,non_streamed_ms,streamed_ms,speedup\n");
        
        int sizes[] = {1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25};
        for (int i = 0; i < 6; i++) {
            int test_n = sizes[i];
            if (test_n > n) break;
            
            // Average over 3 runs for stability
            float time_ns = 0, time_s = 0;
            for (int r = 0; r < 3; r++) {
                time_ns += runNonStreamed(h_A, h_B, h_C, test_n);
                time_s += runStreamed(h_A, h_B, h_C, test_n, S_seg, NUM_STREAMS);
            }
            time_ns /= 3.0f;
            time_s /= 3.0f;
            float speedup = time_ns / time_s;
            
            printf("%d,%.4f,%.4f,%.4f\n", test_n, time_ns, time_s, speedup);
        }
        
        printf("### SEGMENT_SIZE_BENCHMARK ###\n");
        printf("segment_size,non_streamed_ms,streamed_ms,speedup\n");
        
        float baseline_time = 0;
        for (int r = 0; r < 3; r++) {
            baseline_time += runNonStreamed(h_A, h_B, h_C, n);
        }
        baseline_time /= 3.0f;
        
        int segment_sizes[] = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23};
        for (int i = 0; i < 8; i++) {
            int test_seg = segment_sizes[i];
            if (test_seg > n) break;
            
            float time_s = 0;
            for (int r = 0; r < 3; r++) {
                time_s += runStreamed(h_A, h_B, h_C, n, test_seg, NUM_STREAMS);
            }
            time_s /= 3.0f;
            float speedup = baseline_time / time_s;
            
            printf("%d,%.4f,%.4f,%.4f\n", test_seg, baseline_time, time_s, speedup);
        }
        
    } else {
        // Normal mode - single run
        printf("Vector size: %d elements (%.2f MB)\n", n, size / (1024.0 * 1024.0));
        printf("Segment size: %d elements\n", S_seg);
        printf("Number of streams: %d\n\n", NUM_STREAMS);
        
        // Run non-streamed version
        float time_ns = runNonStreamed(h_A, h_B, h_C, n);
        printf("Non-streamed time: %.3f ms\n", time_ns);
        
        // Verify non-streamed result
        vectorAddCPU(h_A, h_B, h_ref, n);
        bool correct_ns = verifyResults(h_ref, h_C, n);
        printf("Non-streamed verification: %s\n\n", correct_ns ? "PASSED" : "FAILED");
        
        // Run streamed version
        float time_s = runStreamed(h_A, h_B, h_C, n, S_seg, NUM_STREAMS);
        printf("Streamed time: %.3f ms\n", time_s);
        
        // Verify streamed result
        bool correct_s = verifyResults(h_ref, h_C, n);
        printf("Streamed verification: %s\n\n", correct_s ? "PASSED" : "FAILED");
        
        // Calculate speedup
        float speedup = time_ns / time_s;
        printf("Speedup: %.2fx\n", speedup);
    }
    
    // Cleanup
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));
    free(h_ref);
    
    return 0;
}
