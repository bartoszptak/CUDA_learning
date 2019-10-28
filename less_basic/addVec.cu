#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
// Defining number of elements in array
#define N 512 * 512 //+1 for error

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
  // Getting index of current kernel
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N) {
    d_c[tid] = d_a[tid] + d_b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main(void) {
  // Declare host and device arrays
  int h_a[N], h_b[N], h_c[N];
  int *d_a, *d_b, *d_c;

  // Allocate Memory on Device
  cudaMalloc((void **)&d_a, N * sizeof(int));
  cudaMalloc((void **)&d_b, N * sizeof(int));
  cudaMalloc((void **)&d_c, N * sizeof(int));
  // Initialize host array
  for (int i = 0; i < N; i++) {
    h_a[i] = 2 * i * i;
    h_b[i] = i;
  }

  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
  // Kernel Call
  gpuAdd<<<512, 512>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
  // This ensures that kernel execution is finishes before going forward
  cudaDeviceSynchronize();
  int Correct = 1;
  printf("Vector addition on GPU \n");
  for (int i = 0; i < N; i++) {
    if ((h_a[i] + h_b[i] != h_c[i])) {
      Correct = 0;
    }
  }
  if (Correct == 1) {
    printf("GPU has computed Sum Correctly\n");
  } else {
    printf("There is an Error in GPU Computation\n");
  }
  // Free up memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
