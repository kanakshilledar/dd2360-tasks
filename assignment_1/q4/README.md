# Question 4 - Rodinia CUDA benchmarks and Comparison with CPU

## Getting Started

We've chosen to run the following benchmarks for this task:

- **CFD**
- **SRAD**

Download the data from the link provided in the assignment description, and place it both `cfd` and `srad` directories under `rodinia_3.1/data` directory.

### CFD

To run the CFD benchmark execute the following commands:

  - **CUDA:**
    ```bash
    cd rodinia_3.1/cuda/cfd
    make
    ./run
    ```

  - **OpenMP:**
    ```bash
    cd rodinia_3.1/openmp/cfd
    make
    ./run
    ```

### SRAD

To run the SRAD benchmark execute the following commands:

  - **CUDA (Compilation):**
    ```bash
    cd rodinia_3.1/cuda/srad
    make
    ```
    - ***Small Input (Running)***
      ```bash
      cd rodinia_3.1/cuda/srad
      ./run
      ```
    - ***Large Input (Running)***
      ```bash
      cd rodinia_3.1/cuda/srad/srad_v1
      ./srad 100 0.5 4096 4096
      ```

  - **OpenMP (Compilation):**
    ```bash
    cd rodinia_3.1/openmp/srad
    make
    ```
    - ***Small Input (Running)***
      ```bash
      cd rodinia_3.1/openmp/srad
      ./run
      ```
    - ***Large Input (Running)***
      ```bash
      cd rodinia_3.1/openmp/srad/srad_v1
      ./srad 100 0.5 4096 4096 8
      ```

