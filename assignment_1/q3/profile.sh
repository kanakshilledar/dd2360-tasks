#!/bin/bash

MATRIX_SIZES=(
  "512 256 256 512"
  "1024 512 512 1024"
  "2048 2048 2048 2048"
  "4096 2048 2048 4096"
  "8192 2048 2048 8192"
  "8192 8192 8192 8192"
  "16384 4096 4096 16384"
)

for SIZE in "${MATRIX_SIZES[@]}"; do
  echo "[+] Profiling with $SIZE"
  # set -- $SIZE
  # A_ROW=$1
  # A_COL=$2
  # B_ROW=$3
  # B_COL=$4

  REPORT_FILE="report_${SIZE}"

  nsys profile -o "$REPORT_FILE" --force-overwrite true ./vecMul $SIZE
done

echo "[*] Profiling complete."

for SIZE in "${MATRIX_SIZES[@]}"; do
  echo "reporting for $SIZE"

  REPORT_FILE="report_${SIZE}.nsys-rep"
  
  nsys stats "$REPORT_FILE" --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum
done
