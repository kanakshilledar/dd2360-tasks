NVCC = nvcc
NVCC_FLAGS = -O3

TARGET = vecAdd

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run normal mode
run: $(TARGET)
	./$(TARGET)

# Run benchmark mode
benchmark: $(TARGET)
	./$(TARGET) --benchmark

# Generate plots (requires Python with matplotlib)
plot: $(TARGET)
	python3 plot_results.py

# Generate nvprof trace for nvvp visualization
profile: $(TARGET)
	nvprof --output-profile profile.nvvp -f ./$(TARGET)
	@echo "Open profile.nvvp with: nvvp profile.nvvp"

# Generate timeline trace for nvvp visualization
timeline: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET)

clean:
	rm -f $(TARGET) *.nvvp *.png

.PHONY: all run benchmark plot profile timeline clean

