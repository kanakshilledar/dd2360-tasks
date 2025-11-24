import matplotlib.pyplot as plt

# Array sizes
sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# CPU times in milliseconds
cpu_times = [0.000977, 0.001953, 0.006104, 0.010986, 0.021973, 0.042969, 0.087158, 0.173096, 0.345947, 0.712891]

# GPU times in milliseconds
gpu_times = [1.155029, 0.044922, 0.047119, 0.187988, 0.065186, 0.032959, 0.085938, 0.107910, 0.230957, 0.358887]

# Bar chart
x = range(len(sizes))
width = 0.35

plt.bar([i - width/2 for i in x], cpu_times, width, label='CPU')
plt.bar([i + width/2 for i in x], gpu_times, width, label='GPU')

plt.xticks(x, sizes, rotation=45)
plt.xlabel('Array size (N)')
plt.ylabel('Time (ms)')
plt.title('CPU vs GPU Reduction Time')
plt.legend()
plt.tight_layout()
plt.show()
