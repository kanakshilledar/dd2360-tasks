#!/bin/bash

VECTOR_SIZES=(
  4096
  8192
  16384
  32768
  65536
  131072
  262144
  524288
  1048576
  2097152
  4194304
  8388608
  16777216
  33554432
  67108864
  134217728
)

for N in "${VECTOR_SIZES[@]}"; do
  echo "Profiling with n = $N"

  REPORT_FILE="report_${N}"
  
  nsys profile -o $REPORT_FILE --force-overwrite true ./vecAdd $N
done

echo "Profiling complete."

for N in "${VECTOR_SIZES[@]}"; do
  echo "reporting for n = $N"

  REPORT_FILE="report_${N}.nsys-rep"
  
  nsys stats $REPORT_FILE --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum
done
