# Assignment III: CUDA Advanced


## Question 1 - Thread Scheduling and Execution Efficienty

From the following kernel:

```cuda
__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) { 

   // Calculate the row # of the d_Pin and d_Pout element to process

 int Row = blockIdx.y*blockDim.y + threadIdx.y;

 // Calculate the column # of the d_Pin and d_Pout element to process 

 int Col = blockIdx.x*blockDim.x + threadIdx.x;

 // each thread computes one element of d_Pout if in range 

 if ((Row < m) && (Col < n)) {

   d_Pout[Row*n+Col] = 2*d_Pin[Row*n+Col];

 }

}
```

Assuming a thread-block size of \(64 \times 16\) threads.

### Case 1: X = 800 , Y = 600

1. **Number of blocks:**

    X-direction:
    $$
    B_x = \left\lceil \frac{800}{64} \right\rceil = 13
    $$

    Y-direction:
    $$
    B_y = \left\lceil \frac{600}{16} \right\rceil = 38
    $$

    Total blocks:
    $$
    B = 13 \times 38 = 494
    $$

2. **Number of warps:**

    Each block has 32 warps, thus the total number of warps is:

    $$
    W = 494 \times 32 = 15808
    $$

3. **Divergent blocks:**

    Remainders:
    $$
    R_x = 800 \bmod 64 = 32
    $$
    $$
    R_y = 600 \bmod 16 = 8
    $$

    Since both remainders are non-zero, divergent blocks occur in:

    - the last column of blocks: \(38\)
    - the last row of blocks: \(13\)
    - substract one overlap:

    $$
    B_d = 38 + 13 - 1 = 50
    $$

4. **Warps with divergence**:

    $$
    W_d = 50 \times 32 = 1600
    $$

### Case 2: X = 600 , Y = 800

1. **Number of blocks:**

    X-direction:
    $$
    B_x = \left\lceil \frac{600}{64} \right\rceil = 10
    $$

    Y-direction:
    $$
    B_y = \left\lceil \frac{800}{16} \right\rceil = 50
    $$

    Total blocks:
    $$
    B = 10 \times 50 = 500
    $$

2. **Number of warps:**

    Each block has 32 warps, thus the total number of warps is:

    $$
    W = 500 \times 32 = 16000
    $$

3. **Divergent blocks:**

    Remainders:
    $$
    R_x = 600 \bmod 64 = 24
    $$
    $$
    R_y = 800 \bmod 16 = 0
    $$

    Since only \(R_x\) is non-zero, divergent blocks occur solely in:

    - the last column of blocks: \(50\)

    Therefore:
    $$
    B_d = 50
    $$

4. **Warps with divergence**:

    $$
    W_d = 50 \times 32 = 1600
    $$


### Case 3: X = 600 , Y = 899

1. **Number of blocks:**

    X-direction:
    $$
    B_x = \left\lceil \frac{600}{64} \right\rceil = 10
    $$

    Y-direction:
    $$
    B_y = \left\lceil \frac{899}{16} \right\rceil = 57
    $$

    Total blocks:
    $$
    B = 10 \times 57 = 570
    $$

2. **Number of warps:**

    Each block has 32 warps, thus the total number of warps is:

    $$
    W = 570 \times 32 = 18240
    $$

3. **Divergent blocks:**

    Remainders:
    $$
    R_x = 600 \bmod 64 = 24
    $$
    $$
    R_y = 899 \bmod 16 = 3
    $$

    Since both remainders are non-zero, divergent blocks occur in:

    - the last column of blocks: \(57\)
    - the last row of blocks: \(10\)
    - subtract one overlap:

    $$
    B_d = 57 + 10 - 1 = 66
    $$

4. **Warps with divergence**:

    $$
    W_d = 66 \times 32 = 2112
    $$


### Summary

| Case  | X   | Y   | Divergent Blocks | Divergent Warps |
| ----- | --- | --- | ---------------- | --------------- |
| **1** | 800 | 600 | 50               | **1600**        |
| **2** | 600 | 800 | 50               | **1600**        |
| **3** | 600 | 899 | 66               | **2112**        |

## Question 2 - CUDA Streams

Using the vecAdd program with CUDA streams to overlap memory transfers and computation. The implementation uses:

- **Kernel**: `C[i] = A[i] + B[i]`
- **Memory**: Pinned (page-locked) host memory
- **Streams**: 4 CUDA streams in round-robin
- **Segment Size**: Configurable (default: 1M elements)

### Performance Comparison

Comparing non-streamed (synchronous) vs streamed (4 streams) execution across vector sizes from 1M to 16M elements:

| Vector Size | Non-Streamed (ms) | Streamed (ms) | Speedup |
|-------------|-------------------|---------------|---------|
| 1M          | 2.23              | 2.11          | 1.05x   |
| 2M          | 4.27              | 3.36          | 1.27x   |
| 4M          | 8.34              | 5.90          | 1.41x   |
| 8M          | 16.70             | 11.00         | 1.52x   |
| 16M         | 32.51             | 21.28         | **1.53x**   |

![Vector Size Comparison](q2/vector_size_comparison.png)

1. **Small vectors (1M)**: Minimal speedup (\(1.05\times\)) due to stream management overhead exceeding overlap benefits.

2. **Large vectors (16M)**: Significant speedup (\(1.53\times\)) from effective overlap of HtoD transfers, kernel execution, and DtoH transfers.

3. **Speedup saturation**: Plateaus at \(\sim1.5\times\) due to PCIe bandwidth limitations and memory-bound nature of the kernel.


### nvvp Visualization

Profiling with nvprof and visualizing in nvvp:

```bash
nvprof --output-profile profile.nvvp -f ./vecAdd
nvvp profile.nvvp
```

**Non-Streamed Execution (Default Stream):**

![Non-Streamed Execution](q2/single_stream.png)

All operations execute sequentially on the default stream with no overlap.

**Streamed Execution (4 Streams):**

![Streamed Execution](q2/streamed.png)

Operations distributed across streams 13-16 with overlapping memory transfers and kernel executions.

In the profiler:
- **MemCpy (HtoD/DtoH) rows**: Continuous transfer activity (olive bars)
- **Compute row**: Kernel executions (cyan bars) interleaved with transfers
- **Stream rows**: Each stream processes segments independently


### Segment Size Impact

Using 16M elements with 4 streams, varying segment size from 64K to 8M:

| Segment Size | Num Segments | Streamed (ms) | Speedup |
|--------------|--------------|---------------|---------|
| 64K          | 256          | 23.12         | 1.42x   |
| 128K         | 128          | 21.50         | 1.52x   |
| 256K         | 64           | 21.11         | 1.55x   |
| 512K         | 32           | 20.97         | **1.56x**   |
| 1M           | 16           | 21.20         | 1.54x   |
| 2M           | 8            | 21.89         | 1.49x   |
| 4M           | 4            | 23.45         | 1.40x   |
| 8M           | 2            | 26.20         | 1.25x   |

![Segment Size Impact](q2/segment_size_impact.png)

1. **Too small (64K)**: 256 segments causes overhead from stream scheduling, reducing speedup to \(1.42\times\).

2. **Optimal (256K-1M)**: Best balance achieving \(\sim1.56\times\) speedup with sufficient parallelism and minimal overhead.

3. **Too large (8M)**: Only 2 segments limits overlap opportunity, speedup drops to \(1.25\times\).

### Summary

| Configuration | Speedup |
|---------------|---------|
| Best vector size | 16M elements |
| Best segment size | 256K-512K elements |
| Maximum speedup | **1.56x** |


## Bonus - A Particle Simulation Application

Implementation of GPU-accelerated particle mover for the iPIC3D-mini particle-in-cell simulation code.

### Environment Setup

1. **Hardware**: NVIDIA GeForce GTX 1650 Ti (Compute Capability 7.5)

2. **Makefile changes**: Updated architecture flag:
   ```
   sm_30 → sm_75
   ```

3. **Build and run**:
   ```bash
   cd bonus/iPIC3D-mini
   make clean && make
   ./bin/miniPIC.out inputfiles/GEM_2D.inp
   ```

### GPU Implementation Design

The `mover_PC_gpu()` function in `Particles_gpu.cu` implements the particle mover on GPU:

1. **Parallelization strategy**: One CUDA thread per particle (\(256\) threads/block)

2. **Memory management**:
   - Flatten 3D field arrays (Ex, Ey, Ez, Bxn, Byn, Bzn) and grid nodes (XN, YN, ZN) to 1D arrays
   - Allocate device memory for particle arrays (x, y, z, u, v, w)
   - Copy data Host→Device before kernel, Device→Host after kernel

3. **Kernel computation**: Each thread performs:
   - Grid-to-particle field interpolation using trilinear weights
   - Boris-style velocity push (handles E and B fields)
   - Position update with subcycling
   - Boundary condition application (periodic/reflecting)

4. **Index mapping**: 3D to 1D flattening via `(ix * Ny + iy) * Nz + iz`

### Correctness Verification

Both CPU and GPU implementations produce the same output files. Numerical comparison using `diff`:

| Output File | Result |
|-------------|--------|
| `E_10.vtk` | **IDENTICAL** |
| `B_10.vtk` | **IDENTICAL** |
| `rhoe_10.vtk` | Equivalent (floating-point rounding only) |
| `rhoi_10.vtk` | Equivalent (floating-point rounding only) |
| `rho_net_10.vtk` | Equivalent (floating-point rounding only) |

**Density file differences** are negligible floating-point rounding errors:
```
CPU: -1.83776e-18    GPU: -1.83775e-18    (diff ≈ 1e-23)
```

**Simulation parameters** (GEM_2D.inp):
- 4 species, \(4{,}096{,}000\) particles each
- Grid: \(256 \times 128 \times 1\) cells
- 10 cycles with \(\Delta t = 0.25\)

### Execution Time Comparison

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Total Simulation Time (s) | 34.11 | 19.97 | **1.71x** |
| Mover Time / Cycle (s) | 1.665 | 0.241 | **6.91x** |
| Interp. Time / Cycle (s) | 1.504 | 1.519 | 0.99x |

1. **Mover kernel speedup**: \(6.91\times\) faster on GPU
   $$
   \text{Speedup}_{\text{mover}} = \frac{1.665}{0.241} \approx 6.91
   $$

2. **Overall simulation speedup**: \(1.71\times\) (limited by CPU-bound interpolation)

3. **Bottleneck**: The `interpP2G()` function remains on CPU, limiting overall speedup according to Amdahl's Law:
   $$
   S = \frac{1}{(1-p) + \frac{p}{s}} \quad \text{where } p \approx 0.53, s \approx 6.91
   $$

### Summary

| Component | CPU Time/Cycle | GPU Time/Cycle | Speedup |
|-----------|----------------|----------------|---------|
| Mover | 1.665s | 0.241s | **6.91x** |
| Interpolation | 1.504s | 1.519s | 0.99x |
| **Total** | **3.41s** | **2.00s** | **1.71x** |


## Attributions

- **Question 1:** Erick Castillo (efce@kth.se)
- **Question 2:** Kanak Shilledar (kanaks@kth.se)
- **Question 3:** Erick Castillo (efce@kth.se) and Kanak Shilledar (kanaks@kth.se)

> **Note:** Both members provided support in Questions 1 and 2
