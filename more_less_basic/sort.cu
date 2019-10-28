#include "device_launch_parameters.h"
#include <stdio.h>

#define arraySize 5
#define threadPerBlock 5
// Kernel Function for Rank sort
__global__ void addKernel(int *d_a, int *d_b) {
  int count = 0;
  int tid = threadIdx.x;
  int ttid = blockIdx.x * threadPerBlock + tid;
  int val = d_a[ttid];
  __shared__ int cache[threadPerBlock];
  
  for (int i = tid; i < arraySize; i += threadPerBlock) {
    cache[tid] = d_a[i];
    __syncthreads();
    for (int j = 0; j < threadPerBlock; ++j)
      if (val > cache[j])
        count++;
    __syncthreads();
  }
  d_b[count] = val;
}

int main() {
  // Define Host and Device Array
  int h_a[arraySize] = {5, 9, 3, 4, 8};
  int h_b[arraySize];
  int *d_a, *d_b;

  // Allocate Memory on the device
  cudaMalloc((void **)&d_b, arraySize * sizeof(int));
  cudaMalloc((void **)&d_a, arraySize * sizeof(int));

  // Copy input vector from host memory to device memory.
  cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

  // Launch a kernel on the GPU with one thread for each element.
  addKernel<<<arraySize / threadPerBlock, threadPerBlock>>>(d_a, d_b);

  // Wait for device to finish operations
  cudaDeviceSynchronize();
  // Copy output vector from GPU buffer to host memory.
  cudaMemcpy(h_b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
  printf("The Enumeration sorted Array is: \n");
  for (int i = 0; i < arraySize; i++) {
    printf("%d\n", h_b[i]);
  }
  // Free up device memory
  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}