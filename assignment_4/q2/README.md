# Heat Equation Solver with cuSPARSE and cuBLAS

1D Heat Equation solver using explicit finite difference method with CUDA libraries (cuSPARSE, cuBLAS) and Unified Memory.

## Build

```bash
make
```

## Run

```bash
./heat <dimX> <nsteps>
```

**Parameters:**
- `dimX` - Grid size (number of points in the rod)
- `nsteps` - Number of time steps to simulate

**Example:**
```bash
./heat 1024 10000
```

## Benchmark & Plot

Run benchmarks and generate convergence plot:
```bash
make plot
```

Or manually:
```bash
python3 plot_results.py
```

This generates `error_vs_nsteps.png` showing relative error vs number of time steps.

## Clean

```bash
make clean
```

