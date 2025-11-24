
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                            \
} while(0)


__global__ void reduction_kernel(int N, float *input, float *result) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) val = input[idx];
    sdata[tid] = val;
    __syncthreads();

    int blockSize = blockDim.x;

    // manually unroll first few steps
    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}


int main(int argc, char **argv) {
  int sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

  for (int i = 0; i < 10; i++) {
    int inputLength = sizes[i];

    //@@ Insert code below to read in inputLength from args
    // inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);
    
    /*@ add other needed data allocation on CPU and GPU here */
    float *h_input = (float*)malloc(inputLength * sizeof(float));
    float *h_result_gpu = (float*)malloc(sizeof(float));

    float *d_input = nullptr;
    float *d_result = nullptr;

    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));  

    //@@ Insert code below to initialize the input array with random values on CPU
    srand(12345);
    for (int i = 0; i < inputLength; i++) {
      h_input[i] = (float)rand() / ((float)RAND_MAX + 1.0f);
    }

    //@@ Insert code below to create reference result in CPU and add a timer
    double cpu_start = get_time_ms();
    float cpu_sum = 0.0f;
    for (int i = 0; i < inputLength; i++)
      cpu_sum += h_input[i];

    double cpu_end = get_time_ms();

    //@@ Insert code to copy data from CPU to the GPU
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    //@@ Initialize the grid and block dimensions here
    int blockSize = 512;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    size_t shmem = blockSize * sizeof(float);

    //@@ Launch the GPU Kernel here and add a timer
    double gpu_start = get_time_ms();
    reduction_kernel<<<gridSize, blockSize, shmem>>>(inputLength, d_input, d_result);
    cudaDeviceSynchronize();
    double gpu_end = get_time_ms();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(h_result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    printf("CPU sum = %f, time = %f\n", cpu_sum, cpu_end - cpu_start);
    printf("GPU sum = %f, time = %f\n", *h_result_gpu, gpu_end - gpu_start);
    printf("Absolute error = %f\n", fabs(cpu_sum - *h_result_gpu));

    // generate csv output

    //@@ Free memory here
    free(h_input);
    free(h_result_gpu);
    cudaFree(d_input);
    cudaFree(d_result);
  }
  return 0;
}

