## Question 1 - Thread Scheduling and Execution Efficienty

From the following kernel:

```c
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

---

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
