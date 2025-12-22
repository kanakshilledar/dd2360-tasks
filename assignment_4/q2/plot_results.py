#!/usr/bin/env python3
"""
Benchmark and plot Heat Equation solver - Relative Error vs nsteps.
Run: python3 plot_results.py
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re

def run_heat_solver(dimX, nsteps):
    """Run the heat solver and extract the relative error."""
    result = subprocess.run(['./heat', str(dimX), str(nsteps)], 
                          capture_output=True, text=True)
    output = result.stdout
    
    # Extract relative error from output
    match = re.search(r'relative error.*is\s+([\d.]+)', output)
    if match:
        return float(match.group(1))
    return None

def main():
    print("Running Heat Equation benchmarks...")
    
    # Question 2: Error vs nsteps for dimX=1024
    dimX = 1024
    nsteps_values = [100, 200, 500, 1000, 2000, 5000, 10000]
    errors = []
    
    print(f"\n{'nsteps':>10} {'Relative Error':>16}")
    print("-" * 30)
    
    for nsteps in nsteps_values:
        error = run_heat_solver(dimX, nsteps)
        errors.append(error)
        print(f"{nsteps:>10} {error:>16.6f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nsteps_values, errors, 'o-', color='#2563eb', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Time Steps (nsteps)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title(f'Heat Equation Convergence (dimX={dimX})', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Use exact numbers instead of powers of 10
    ax.set_xticks(nsteps_values)
    ax.set_xticklabels([str(n) for n in nsteps_values])
    ax.set_yticks([3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax.set_yticklabels(['3', '4', '5', '6', '7', '8', '9', '10', '11'])
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_nsteps.png', dpi=150)
    print("\nSaved: error_vs_nsteps.png")
    plt.close()

if __name__ == '__main__':
    main()
