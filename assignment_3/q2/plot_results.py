#!/usr/bin/env python3
"""
Benchmark and plot CUDA stream performance for vector addition.
Run: python3 plot_results.py
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark():
    """Run the CUDA benchmark and parse results."""
    result = subprocess.run(['./vecAdd', '--benchmark'], capture_output=True, text=True)
    output = result.stdout
    
    vector_data = {'sizes': [], 'non_streamed': [], 'streamed': [], 'speedup': []}
    segment_data = {'sizes': [], 'non_streamed': [], 'streamed': [], 'speedup': []}
    
    current_section = None
    
    for line in output.strip().split('\n'):
        if 'VECTOR_SIZE_BENCHMARK' in line:
            current_section = 'vector'
            continue
        elif 'SEGMENT_SIZE_BENCHMARK' in line:
            current_section = 'segment'
            continue
        elif line.startswith('vector_size') or line.startswith('segment_size'):
            continue  # Skip header
        elif ',' in line and current_section:
            parts = line.split(',')
            if len(parts) == 4:
                size, ns, s, speedup = parts
                data = vector_data if current_section == 'vector' else segment_data
                data['sizes'].append(int(size))
                data['non_streamed'].append(float(ns))
                data['streamed'].append(float(s))
                data['speedup'].append(float(speedup))
    
    return vector_data, segment_data

def plot_vector_size_comparison(data):
    """Plot performance comparison at different vector lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = np.array(data['sizes'])
    size_labels = [f'{s/(1<<20):.0f}M' if s >= (1<<20) else f'{s/(1<<10):.0f}K' for s in sizes]
    x = np.arange(len(sizes))
    width = 0.35
    
    # Execution time comparison
    bars1 = ax1.bar(x - width/2, data['non_streamed'], width, label='Non-Streamed', color='#e74c3c')
    bars2 = ax1.bar(x + width/2, data['streamed'], width, label='Streamed (4 streams)', color='#3498db')
    
    ax1.set_xlabel('Vector Size (elements)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Vector Addition: Streamed vs Non-Streamed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Speedup plot
    ax2.plot(x, data['speedup'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Vector Size (elements)')
    ax2.set_ylabel('Speedup (Non-Streamed / Streamed)')
    ax2.set_title('Speedup from CUDA Streams')
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vector_size_comparison.png', dpi=150)
    print("Saved: vector_size_comparison.png")
    plt.close()

def plot_segment_size_impact(data):
    """Plot impact of segment size on performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = np.array(data['sizes'])
    size_labels = [f'{s/(1<<10):.0f}K' if s < (1<<20) else f'{s/(1<<20):.0f}M' for s in sizes]
    x = np.arange(len(sizes))
    
    # Execution time vs segment size
    ax1.plot(x, data['non_streamed'], 's--', color='#e74c3c', linewidth=2, markersize=8, label='Non-Streamed (baseline)')
    ax1.plot(x, data['streamed'], 'o-', color='#3498db', linewidth=2, markersize=8, label='Streamed')
    
    ax1.set_xlabel('Segment Size (elements)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Impact of Segment Size on Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, rotation=45)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Speedup vs segment size
    ax2.bar(x, data['speedup'], color='#9b59b6', alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Segment Size (elements)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Segment Size (Vector: 16M elements)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels, rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('segment_size_impact.png', dpi=150)
    print("Saved: segment_size_impact.png")
    plt.close()

def main():
    print("Running CUDA benchmark...")
    vector_data, segment_data = run_benchmark()
    
    if vector_data['sizes']:
        print("\n=== Vector Size Benchmark Results ===")
        print(f"{'Size':>12} {'Non-Streamed':>14} {'Streamed':>12} {'Speedup':>10}")
        for i in range(len(vector_data['sizes'])):
            print(f"{vector_data['sizes'][i]:>12} {vector_data['non_streamed'][i]:>14.3f} {vector_data['streamed'][i]:>12.3f} {vector_data['speedup'][i]:>10.3f}x")
        plot_vector_size_comparison(vector_data)
    
    if segment_data['sizes']:
        print("\n=== Segment Size Benchmark Results ===")
        print(f"{'Segment':>12} {'Non-Streamed':>14} {'Streamed':>12} {'Speedup':>10}")
        for i in range(len(segment_data['sizes'])):
            print(f"{segment_data['sizes'][i]:>12} {segment_data['non_streamed'][i]:>14.3f} {segment_data['streamed'][i]:>12.3f} {segment_data['speedup'][i]:>10.3f}x")
        plot_segment_size_impact(segment_data)
    
    print("\nDone! Check the generated PNG files.")

if __name__ == '__main__':
    main()

