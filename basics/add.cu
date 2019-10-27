#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) 
{
   *d_c = *d_a + *d_b;
}

int main() 
{
   int h_a, h_b, h_c;
   int *d_a, *d_b, *d_c;

   h_a = 1;
   h_b = 4;

   cudaMalloc((void**)&d_a, sizeof(int));
   cudaMalloc((void**)&d_b, sizeof(int));
   cudaMalloc((void**)&d_c, sizeof(int));

   cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

   gpuAdd<<<1,1>>>(d_a, d_b, d_c);

   cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
   printf("Passing Parameter by Reference Output: %d + %d = %d\n", h_a, h_b, h_c);

   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
   return 0;
}