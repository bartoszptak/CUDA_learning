#include <iostream>

__global__ void myfirstkernel() {}

int main() {
  myfirstkernel<<<1, 1>>>();
  printf("Hello world");
  return 0;
}