#!/bin/bash

# This script finds all 'report_*.nsys-rep' files, extracts the three
# key timing values, and prints them in CSV format.
#
# Run it like this:
# ./parse_reports.sh > results.csv

# Print the CSV header line. This will be the first line of your output.
echo "VectorSize,HtoD_Time_ns,Kernel_Time_ns,DtoH_Time_ns"

# Loop over all nsys-rep files in the current directory
for file in report_*.nsys-rep; do
  
  # This line protects against the loop running if no files match
  [ -e "$file" ] || { echo "Error: No report_*.nsys-rep files found." >&2; exit 1; }
  
  # Extract the vector size 'n' from the filename (e.g., "report_1048576.nsys-rep" -> "1048576")
  n=$(basename "$file" .nsys-rep | cut -d_ -f2)
  
  # Print progress to the terminal (stderr) so it doesn't go into the CSV
  echo "Processing $file (n=$n)..." >&2
  
  # Run nsys stats and pipe the output directly to awk for parsing
  # awk is a powerful tool for finding lines and extracting columns
  nsys stats "$file" --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum | awk -v n="$n" '
    
    # Find the HtoD time line. We must escape the [ and ] brackets for regex
    /\[CUDA memcpy Host-to-Device\]/ {
        htod = $2;        # Get the 2nd column (Total Time)
        gsub(",", "", htod); # Remove commas from the number (e.g., 2,442,397 -> 2442397)
    }

    # Find the DtoH time line
    /\[CUDA memcpy Device-to-Host\]/ {
        dtoh = $2;
        gsub(",", "", dtoh);
    }

    # Find the kernel time line (we just look for "vectorAdd")
    /vectorAdd/ {
        kernel = $2;
        gsub(",", "", kernel);
    }

    # At the very end of processing the output, print our formatted CSV line
    END {
        print n "," (htod+0) "," (kernel+0) "," (dtoh+0);
    }
  '
done

echo "Parsing complete. CSV data has been printed to standard output." >&2
