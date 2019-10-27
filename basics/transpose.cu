#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
 //Defining number of elements in Array
#define N 5
#define M 6
//Kernel function for squaring number

__global__ void gpuTranspose(float *d_in, float *d_out) 
{
     //Getting thread index for current kernel
     int tid = threadIdx.x; // handle the data at this index
     int bid = blockIdx.x;
     (*d_out)[bid][tid] = (*d_in)[tid][bid];
 }


int main(void) 
{

     float h_in[N][M], h_out[M][N];
     float *d_in, *d_out;

     cudaMalloc((void**)&d_in, N * M * sizeof(float));
     cudaMalloc((void**)&d_out, N * M * sizeof(float));

     for (int i = 0; i < N; i++) 
     {
         for (int j = 0; j < M; j++){
            h_in[i][j] = i*N+j;
         }
     }

    cudaMemcpy(d_in, h_in, N * M * sizeof(float), cudaMemcpyHostToDevice);
    gpuTranspose << <N, M >> >(d_in, d_out);
    cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

 
     printf("Array \n");
     for (int i = 0; i < N; i++) 
     {
         for(int j = 0; j < M; j++){
            printf("%f\t", h_in[i][j]);
         }
         printf("\n");
     }

     printf("Transposed rray \n");
     for (int i = 0; i < M; i++) 
     {
         for(int j = 0; j < N; j++){
            printf("%f\t", h_out[i][j]);
         }
         printf("\n");
     }

     cudaFree(d_in);
     cudaFree(d_out);

     return 0;
    }
