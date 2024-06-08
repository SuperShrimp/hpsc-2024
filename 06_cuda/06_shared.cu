#include <cstdio>

__device__ __managed__ int sum;

__global__ void reduction(int &sum, int *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int b[];
  __syncthreads();
  //Each thread loads to it's part of shared memory
  //每个bock中有对应的的threadidx
  b[threadIdx.x] = a[i];
  __syncthreads();
  int c = 0;
  //对每个block执行一次summary
  for (int j=0; j<blockDim.x; j++)
    c += b[j];
  //one thread per block
  if (threadIdx.x == 0)
    atomicAdd(&sum, c);
}

int main(void) {
  const int N = 128;
  const int M = 64;
  int *a;
  cudaMallocManaged(&a, N*sizeof(int));
  for (int i=0; i<N; i++) a[i] = 1;
  //Dynamic allocation of shared memory
  //第三个参数是线程块需要的大小。
  reduction<<<N/M,M,M*sizeof(int)>>>(sum, a);
  cudaDeviceSynchronize();
  printf("%d\n",sum);
  cudaFree(a);
}
