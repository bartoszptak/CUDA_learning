#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#define N 1024
#define threadsPerBlock 512

__global__ void gpu_dot(float *d_a, float *d_b, float *d_c) {
  // Define Shared Memory
  __shared__ float partial_sum[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int index = threadIdx.x;

  float sum = 0;
  while (tid < N) {
    sum += d_a[tid] * d_b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // set the partial sum in shared memory
  partial_sum[index] = sum;

  // synchronize threads in this block
  __syncthreads();

  // Calculate Patial sum for a current block using data in shared memory
  int i = blockDim.x / 2;
  while (i != 0) {
    if (index < i) {
      partial_sum[index] += partial_sum[index + i];
    }
    __syncthreads();
    i /= 2;
  }
  // Store result of partial sum for a block in global memory
  if (index == 0)
    d_c[blockIdx.x] = partial_sum[0];
}

int main(void) {
  float *h_a, *h_b, h_c, *partial_sum;
  float *d_a, *d_b, *d_partial_sum;

  // Calculate number of blocks and number of threads
  int block_calc = (N + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGrid = (32 < block_calc ? 32 : block_calc);
  // allocate memory on the cpu side
  h_a = (float *)malloc(N * sizeof(float));
  h_b = (float *)malloc(N * sizeof(float));
  partial_sum = (float *)malloc(blocksPerGrid * sizeof(float));

  // allocate the memory on the gpu
  cudaMalloc((void **)&d_a, N * sizeof(float));
  cudaMalloc((void **)&d_b, N * sizeof(float));
  cudaMalloc((void **)&d_partial_sum, blocksPerGrid * sizeof(float));

  // fill in the host mempory with data
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = 2;
  }

  // copy the arrays to the device
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  gpu_dot<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_partial_sum);

  // copy the array back to the host
  cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Calculate final dot prodcut
  h_c = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    h_c += partial_sum[i];
  }

  printf("The computed dot product is: %f\n", h_c);
  #define cpu_sum(x) (x * (x + 1))
  if (h_c == cpu_sum((float)(N - 1))) {
    printf("The dot product computed by GPU is correct\n");
  } else {
    printf("Error in dot product computation");
  }
  // free memory on the gpu side
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_partial_sum);
  // free memory on the cpu side
  free(h_a);
  free(h_b);
  free(partial_sum);
}