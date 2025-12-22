NVCC = nvcc
NVCC_FLAGS = -O3
LIBS = -lcusparse -lcublas

TARGET = heat

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

run: $(TARGET)
	./$(TARGET) 1024 10000

plot: $(TARGET)
	python3 plot_results.py

clean:
	rm -f $(TARGET) *.png

.PHONY: all run plot clean
