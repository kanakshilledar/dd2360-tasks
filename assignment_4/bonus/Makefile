# Makefile for Tensor Core WMMA GEMM
# Requires SM 7.0+ for WMMA API

NVCC = nvcc
ARCH = sm_75
NVCC_FLAGS = -arch=$(ARCH) -O3
TARGET = wmma_gemm
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) 1024 2048 2048 1024

plot: $(TARGET)
	python3 plot.py

clean:
	rm -f $(TARGET) *.png

.PHONY: all clean run plot
