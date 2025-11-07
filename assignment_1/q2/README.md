# Q2: Vector Addition

To run the code following the following instructions

```sh
nvcc vecAdd.cu -o vecAdd
./vecAdd <size of array>
```

## Instructions to profile
There is a helper script which will try to profile the code for you using `nsys` tool.

```sh
./profil.sh
```

This will give you the required memory operations and kernel time.