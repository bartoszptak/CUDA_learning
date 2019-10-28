#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void handle_errors(void){
    int *d_a;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_a, sizeof(int));
    printf("Status: %d, cudaSuccess: %d\n", cudaStatus, cudaSuccess);

    int *h_a;
    cudaStatus = cudaMemcpy(d_a,&h_a, sizeof(int), cudaMemcpyHostToDevice);
    printf("Status: %d, cudaSuccess: %d\n", cudaStatus, cudaSuccess);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(d_a);
        }

    cudaStatus = cudaGetLastError();
    printf("Status: %d, cudaSuccess: %d\n", cudaStatus, cudaSuccess);
    
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
}

void properties(void){
    cudaDeviceProp device_Property;
    cudaGetDeviceProperties(&device_Property, 0);
    printf("Max per procesor: %d\n", device_Property.maxThreadsPerMultiProcessor);
    printf("Max per block: %d\n", device_Property.maxThreadsPerBlock);
    printf("CUDA stream: %d\n", device_Property.deviceOverlap);
}

int main(void){
    handle_errors();
    properties();
    

    return 0;
}