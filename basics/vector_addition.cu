#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 100000

// Test result
// No Size    GPU       CPU
// 1  10      0.098583  0.000047
// 2  10000   0.106036  0.000428
// 3  100000  0.102938  0.003612

// Defining vector addition function for CPU
void cpuAdd(int *h_a, int *h_b, int *h_c) {
  int tid = 0;
  while (tid < N) {
    h_c[tid] = h_a[tid] + h_b[tid];
    tid += 1;
  }
}

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
  // Getting block index of current kernel
  int tid = blockIdx.x; // handle the data at this index
  if (tid < N)
    d_c[tid] = d_a[tid] + d_b[tid];
}

void runCPU() {
  int h_a[N], h_b[N], h_c[N];
  for (int i = 0; i < N; i++) {
    h_a[i] = 2 * i * i;
    h_b[i] = i;
  }

  // Calling CPU function for vector addition
  cpuAdd(h_a, h_b, h_c);
  // Printing Answer
  printf("Vector addition on CPU\n");
  // for (int i = 0; i < N; i++) {
  //   printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i],
  //          h_c[i]);
  // }
}

void runGPU() {
  int h_a[N], h_b[N], h_c[N];
  int *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, N * sizeof(int));
  cudaMalloc((void **)&d_b, N * sizeof(int));
  cudaMalloc((void **)&d_c, N * sizeof(int));

  for (int i = 0; i < N; i++) {
    h_a[i] = 2 * i * i;
    h_b[i] = i;
  }

  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

  gpuAdd<<<N,1>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Vector addition on GPU \n");
  //Printing result on console
  // for (int i = 0; i < N; i++) 
  // {
  //     printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i],             h_c[i]);
  // }
  //Free up memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main(void) {

  clock_t start_h = clock();
  runCPU();
  clock_t end_h = clock();
  double time_h = (double)(end_h - start_h) / CLOCKS_PER_SEC;

  clock_t start_d = clock();
  runGPU();
  cudaThreadSynchronize();
  clock_t end_d = clock();
  double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;
  printf("No of Elements in Array:%d \n Device time %f seconds \n host time %f Seconds\n", N, time_d, time_h);

  return 0;
}