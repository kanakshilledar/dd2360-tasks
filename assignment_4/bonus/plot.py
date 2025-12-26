#!/usr/bin/env python3
"""
Plot benchmark results for WMMA Tensor Core GEMM
Question 4: Compare CPU, GEMM, and WMMA at 5 matrix sizes
"""

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re

def run_benchmark(size):
    """Run wmma_gemm and parse output"""
    cmd = f"./wmma_gemm {size} {size} {size} {size}"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    output = result.stdout
    
    # Parse timing values
    cpu_match = re.search(r"CPU:\s+([\d.]+)\s*ms", output)
    gemm_match = re.search(r"GEMM:\s+([\d.]+)\s*ms", output)
    wmma_match = re.search(r"WMMA:\s+([\d.]+)\s*ms.*maxErr=([\d.]+).*avgErr=([\d.]+)", output)
    
    cpu_ms = float(cpu_match.group(1)) if cpu_match else 0
    gemm_ms = float(gemm_match.group(1)) if gemm_match else 0
    wmma_ms = float(wmma_match.group(1)) if wmma_match else 0
    max_err = float(wmma_match.group(2)) if wmma_match else 0
    avg_err = float(wmma_match.group(3)) if wmma_match else 0
    
    return cpu_ms, gemm_ms, wmma_ms, max_err, avg_err

def main():
    sizes = [512, 1024, 2048, 4096, 8192]
    
    cpu_times = []
    gemm_times = []
    wmma_times = []
    max_errors = []
    avg_errors = []
    
    print("Running benchmarks...")
    for size in sizes:
        print(f"  {size}x{size}...", end=" ", flush=True)
        cpu, gemm, wmma, max_err, avg_err = run_benchmark(size)
        cpu_times.append(cpu)
        gemm_times.append(gemm)
        wmma_times.append(wmma)
        max_errors.append(max_err)
        avg_errors.append(avg_err)
        print(f"CPU={cpu:.1f}ms, GEMM={gemm:.1f}ms, WMMA={wmma:.1f}ms")
    
    # Plot 1: Runtime comparison bar chart
    x = np.arange(len(sizes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, cpu_times, width, label='CPU', color='#3498db')
    bars2 = ax.bar(x, gemm_times, width, label='GEMM', color='#2ecc71')
    bars3 = ax.bar(x + width, wmma_times, width, label='WMMA', color='#e74c3c')
    
    ax.set_xlabel('Matrix Size (NxN)')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Matrix Multiplication: CPU vs GEMM vs WMMA Tensor Core')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('runtime_comparison.png', dpi=150)
    plt.savefig('runtime_comparison.pdf')
    print("\nSaved: runtime_comparison.png")
    
    # Plot 2: Speedup comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    gemm_speedup = [c/g for c, g in zip(cpu_times, gemm_times)]
    wmma_speedup = [c/w for c, w in zip(cpu_times, wmma_times)]
    
    ax.bar(x - width/2, gemm_speedup, width, label='GEMM vs CPU', color='#2ecc71')
    ax.bar(x + width/2, wmma_speedup, width, label='WMMA vs CPU', color='#e74c3c')
    
    ax.set_xlabel('Matrix Size (NxN)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('GPU Speedup over CPU')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('speedup.png', dpi=150)
    plt.savefig('speedup.pdf')
    print("Saved: speedup.png")
    
    # Plot 3: Accuracy loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, max_errors, width, label='Max Error', color='#e74c3c')
    ax.bar(x + width/2, avg_errors, width, label='Avg Error', color='#3498db')
    
    ax.set_xlabel('Matrix Size (NxN)')
    ax.set_ylabel('Error (absolute)')
    ax.set_title('WMMA Accuracy Loss (half precision vs float)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('accuracy.png', dpi=150)
    plt.savefig('accuracy.pdf')
    print("Saved: accuracy.png")
    
    # Print summary table
    print("\n" + "="*70)
    print(f"{'Size':<10} {'CPU (ms)':<12} {'GEMM (ms)':<12} {'WMMA (ms)':<12} {'WMMA Err':<12}")
    print("="*70)
    for i, size in enumerate(sizes):
        print(f"{size}x{size:<4} {cpu_times[i]:<12.2f} {gemm_times[i]:<12.2f} {wmma_times[i]:<12.2f} {avg_errors[i]:<12.6f}")
    print("="*70)

if __name__ == "__main__":
    main()

