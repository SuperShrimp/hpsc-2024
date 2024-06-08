#include <cstdio>

__device__ __managed__ int sum;

__global__ void kernel(int &sum, int *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //用AtomicAdd直接把所有元素合并到sum里面
  atomicAdd(&sum, a[i]);
}

int main(void) {
  const int N = 128;
  const int M = 64;
  int *a;
  cudaMallocManaged(&a, N*sizeof(int));
  for (int i=0; i<N; i++) a[i] = 1;
  kernel<<<N/M,M>>>(sum, a);
  cudaDeviceSynchronize();
  printf("%d\n",sum);
  cudaFree(a);
}
