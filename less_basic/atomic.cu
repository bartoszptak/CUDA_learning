#include <stdio.h>

#define NUM_THREADS 10000
#define SIZE 10
#define BLOCK_WIDTH 100

__global__ void gpu_increment_without_atomic(int *d_a)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread increment elements which wraps at SIZE
  tid = tid % SIZE;
  d_a[tid] += 1;
}

__global__ void gpu_increment_atomic(int *d_a)
{
  // Calculate thread index 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread increments elements which wraps at SIZE
  tid = tid % SIZE;
  atomicAdd(&d_a[tid], 1);
}

void runWithout(void){
    printf("%d total threads in %d blocks writing into %d array elements\n",
  NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

  // declare and allocate host memory
  int h_a[SIZE];
  const int ARRAY_BYTES = SIZE * sizeof(int);
  // declare and allocate GPU memory
  int * d_a;
  cudaMalloc((void **)&d_a, ARRAY_BYTES);

  // Initialize GPU memory with zero value.
  cudaMemset((void *)d_a, 0, ARRAY_BYTES);
  gpu_increment_without_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);

  // copy back the array of sums from GPU and print
  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  printf("Number of times a particular Array index has been incremented without atomic add is: \n");
  for (int i = 0; i < SIZE; i++)
  {
    printf("index: %d --> %d times\n ", i, h_a[i]);
  }
  cudaFree(d_a);

}

void runWith(void){
    printf("%d total threads in %d blocks writing into %d array elements\n",NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

  // declare and allocate host memory
  int h_a[SIZE];
  const int ARRAY_BYTES = SIZE * sizeof(int);

  // declare and allocate GPU memory
  int * d_a;
  cudaMalloc((void **)&d_a, ARRAY_BYTES);

   // Initialize GPU memory withzero value
  cudaMemset((void *)d_a, 0, ARRAY_BYTES);

  gpu_increment_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);
    // copy back the array from GPU and print
  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  printf("Number of times a particular Array index has been incremented is: \n");
  for (int i = 0; i < SIZE; i++) 
  { 
     printf("index: %d --> %d times\n ", i, h_a[i]); 
  }

  cudaFree(d_a);
}

int main(void){
    runWithout();
    runWith();
}