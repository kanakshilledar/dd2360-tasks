#import "@preview/tablem:0.3.0": tablem, three-line-table
#set page(
  paper: "a4",
  header: "DD2360 | Group 29",
  numbering: "1",
)

#title[Assignment 2: CUDA Basics]

= Q[1]

= Q[2] Reduction
== 1. Describe all optimizations you attempted, and whether they improved or hurt performance
=== Changing the block from the default 256
For reference the output with value of `N = 1000000`
- Block Size of 256.
```
The input length is 1000000
CPU sum = 499877.125000, time = 2.650879
GPU sum = 499877.000000, time = 1.864014
Absolute error = 0.125000
```
- Block Size of 128
```
The input length is 1000000
CPU sum = 499877.125000, time = 2.746826
GPU sum = 499876.750000, time = 1.531982
Absolute error = 0.375000
```
- Block Size of 512
```
The input length is 1000000
CPU sum = 499877.125000, time = 2.683105
GPU sum = 499877.156250, time = 1.753906
Absolute error = 0.031250
```
- Block Size of 1024
```
The input length is 1000000
CPU sum = 499877.125000, time = 2.772217
GPU sum = 499877.093750, time = 1.918945
Absolute error = 0.031250
```

=== Asynchronous memcopy
Having asynchronous memcopy can be beneficial if there are more than 1 kernels. Having just a single kernel in this case makes this optimization not useful enough to be able to provide valuable results.

=== Loop Unrolling
Loop unrolling manually expands the first few iterations of a loop to reduce the overhead of loop control and branch divergence.
For block size of 512 with loop unrolling we get
```
The input length is 1000000
CPU sum = 499877.125000, time = 2.656982
GPU sum = 499877.125000, time = 1.750977
Absolute error = 0.000000
```

== 2. Which optimizations did you choose in the end and why?
We went ahead with increasing the block size to 512 and perform loop unrolling. This combination helped us achieve good optimization. Having a higher block size helps in improving the utilization of the GPU and reduces the total number of atomic operations. 

== 3. How many global memory reads and writes are performed in total in one run? Explain your answer.
Each threads reads one element from the input array, so for N elements it will be N reads. Each block writes one value to the global memory, so for N elements, number of writes will be $ ceil(N /"blockSize") $

== 4. Do you use atomic operations? If so, how many atomic operations are performed in total? Explain your answer.
Atomic Add operation is used to update the global result. These operations are specifically used in case of shared memory. One atomic add operation is performed per block, by the first thread of each block.
Thus for N elements,
$ "number of atomic operations" = ceil(N / "blockSize") $

== 5. Do you use atomic operations? If so, how much shared memory is used in your code? Explain your answer.
The kernel uses `atomicAdd()` operation to write the partial sum to global shared memory.
The shared memory array holds all elements of the block loaded from the global memory. This makes the size of shared memory to be
$ "size" = "blockSize" * "sizeof(<datatype>)" $

== 6. Run your program on array lengths starting from 512 and increasing at x2 rate for 8-10 runs. Plot a bar chart to visualize the time taken by CPU and GPU for each run.

After running the plotting script, we get the following chart.
#figure(
  image("asn2_q2.svg"),
  caption: [
    CPU vs GPU time of for various size of arrays.
  ],
)

== 7. Do you observe any speed from using GPU? Why or why not?
After observing the graph, we can conclude that for smaller array sizes upto 8192, the GPU computation time is slower than that of CPU. This is because for smaller value of N, the kernel launch and memory transfer overhead is much more than just running it on the CPU. In case of larger values, this overhead becomes negligible.

== 8. Using the largest array size in 6, profile and report Shared Memory Configuration Size and Achieved Occupancy. Describe one optimization that you would like to try.

