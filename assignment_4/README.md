# Assignment IV: NVIDIA Libraries and Unified Memory

## Question 2 - Heat Equation Solver with cuSPARSE and cuBLAS

Implementation of an explicit finite difference method for the 1D Heat Equation using cuSPARSE, cuBLAS, and Unified Memory.

### FLOPS Analysis at Different Grid Sizes

**FLOPS Calculation Method:**

For each iteration, the algorithm performs:
1. **SpMV** (Sparse Matrix-Vector Multiplication): `tmp = A * temp`
   - Operations: `2 x nzv` (multiply + add per non-zero)
   - Where `nzv = 3 x dimX - 6` (tridiagonal matrix)

2. **AXPY**: `temp = α × tmp + temp`
   - Operations: `2 × dimX` (multiply + add per element)

3. **Norm**: `||tmp||`
   - Operations: `2 × dimX` (square + accumulate per element)

**FLOPS per iteration:**
$$
\text{FLOPS}_{\text{iter}} = 2(3 \cdot \text{dimX} - 6) + 2 \cdot \text{dimX} + 2 \cdot \text{dimX} = 10 \cdot \text{dimX} - 12
$$

**Results (10,000 iterations):**

| dimX | nzv | FLOPS/iter | Total FLOPS | Time (s) | GFLOPS |
|------|-----|------------|-------------|----------|--------|
| 256  | 762 | 2,548 | 2.55×10⁷ | ~2.5 | ~0.01 |
| 512  | 1,530 | 5,108 | 5.11×10⁷ | ~2.6 | ~0.02 |
| 1024 | 3,066 | 10,228 | 1.02×10⁸ | ~2.6 | ~0.04 |
| 2048 | 6,138 | 20,468 | 2.05×10⁸ | ~3.3 | ~0.06 |
| 4096 | 12,282 | 40,948 | 4.09×10⁸ | ~3.9 | ~0.10 |

**Observations:**
- GFLOPS is relatively low because the algorithm is memory-bound rather than compute-bound
- The sparse matrix operations have low arithmetic intensity (few FLOPs per byte transferred)
- cuBLAS/cuSPARSE library overhead dominates for small problem sizes

---

### Convergence Analysis - Error vs Time Steps

Running with `dimX=1024` and varying `nsteps` from 100 to 10,000:

| nsteps | Relative Error |
|--------|----------------|
| 100    | 10.186 |
| 200    | 8.677 |
| 500    | 6.947 |
| 1000   | 5.830 |
| 2000   | 4.859 |
| 5000   | 3.770 |
| 10000  | 3.067 |

![Convergence Plot](q2/error_vs_nsteps.png)

**Observations:**

1. **Slow convergence**: The relative error decreases slowly with increasing iterations. Even at 10,000 steps, the error remains above 3.0.

2. **Power-law decay**: On a log-log scale, the error decreases approximately linearly, indicating a power-law relationship: `error ∝ nsteps^(-α)` with α ≈ 0.25.

3. **Explicit method limitation**: The explicit finite difference method for parabolic PDEs (like the heat equation) converges slowly. This is expected because:
   - The method has CFL stability constraint: `α × Δt / Δx² ≤ 0.5`
   - Each iteration only propagates information by one grid cell
   - For dimX=1024, information must propagate ~500 cells to reach steady state

4. **Practical implication**: For low error (<1%), many more iterations are needed (e.g., 100,000+ steps for ~1% error, 2,000,000+ steps for ~0.1% error).

---

### Prefetching Impact on Unified Memory Performance

**Note:** TODO

**Theoretical Analysis:**

With prefetching enabled:
- Data is explicitly migrated to CPU before initialization
- Data is explicitly migrated to GPU before computation
- Reduces page faults during kernel execution

Without prefetching:
- Unified Memory uses on-demand page migration
- Page faults occur when GPU first accesses each page
- Can cause significant latency during first kernel executions

**Expected Performance Impact:**

| Scenario | Page Faults | First Iteration | Subsequent Iterations |
|----------|-------------|-----------------|----------------------|
| With Prefetch | Minimal | Fast | Fast |
| Without Prefetch | Many | Slow (page faults) | Fast (data cached) |

For iterative algorithms like this heat equation solver:
- Prefetching provides **most benefit on first few iterations**
- After initial migration, both approaches perform similarly
- Overall speedup depends on ratio of initialization to computation time

**Profiling with ncu (optional):**
```bash
ncu --set full -o heat_profile ./heat 1024 10000
```

Metrics to observe:
- `gpu__dram_throughput.avg.pct_of_peak_sustained_active`
- `lts__t_sectors_srcunit_tex_aperture_peer_lookup_miss.sum`
- `sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`

