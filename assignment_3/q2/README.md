# CUDA Streams Vector Addition

Vector addition using CUDA streams to overlap memory transfers and computation.

## Build

```bash
make
```

## Run

**Basic run** (16M elements, 1M segment size, 4 streams):
```bash
./vecAdd
```

**Custom parameters**:
```bash
./vecAdd -n <vector_size> -s <segment_size> --streams <num_streams>
```

Example:
```bash
./vecAdd -n 33554432 -s 524288 --streams 4
```

## Benchmark & Plot

Run benchmarks and generate plots:
```bash
python3 plot_results.py
```

Or using make:
```bash
make benchmark   # Run benchmark only
make plot        # Run benchmark and generate plots
```

## Profiling

Generate nvprof trace for nvvp:
```bash
make profile
nvvp profile.nvvp
```

Print GPU trace to terminal:
```bash
make timeline
```

## Clean

```bash
make clean
```

