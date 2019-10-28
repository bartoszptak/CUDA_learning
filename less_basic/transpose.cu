#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
// Defining number of elements in Array
#define N 5
#define M 6
#define BLOCK_SIZE 512
// Kernel function for squaring number

__global__ void gpuTranspose(float *d_in, float *d_out, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < cols && idy < rows) {
    unsigned int pos = idy * cols + idx;
    unsigned int trans_pos = idx * rows + idy;
    d_out[trans_pos] = d_in[pos];
  }
}

int main(void) {

  float h_in[N][M], h_out[M][N];
  float *d_in, *d_out;

  cudaMalloc((void **)&d_in, N * M * sizeof(float));
  cudaMalloc((void **)&d_out, N * M * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      h_in[i][j] = i * N + j;
    }
  }

  cudaMemcpy(d_in, h_in, N * M * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
  gpuTranspose<<<dim_block, dim_grid>>>(d_in, d_out, N, M);
  cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Array \n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%f\t", h_in[i][j]);
    }
    printf("\n");
  }

  printf("Transposed array \n");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f\t", h_out[i][j]);
    }
    printf("\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
