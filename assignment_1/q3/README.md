# Q3: Vector Multiplication

To run the code following the following instructions

```sh
nvcc vecMul.cu -o vecMul
nvcc vecMulDouble.cu -o vecMulDouble
./vecMul <size of array>
./vecMulDouble <size of array>
```

## Instructions to profile
There is a helper script which will try to profile the code for you using `nsys` tool.

```sh
./profile.sh
```

This will give you the required memory operations and kernel time.