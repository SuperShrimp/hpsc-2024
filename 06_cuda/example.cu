#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <ctime>   // For time()
#include <cuda_runtime.h>

__global__ void count_kernel(int *d_key, int *d_bucket, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&d_bucket[d_key[idx]], 1);
        __syncthreads();
    }
}

__global__ void sort_kernel(int *d_key, int *d_bucket, int n, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < range) {
        int count = d_bucket[idx];
        __syncthreads();
        for (int i = 0; i < count; i++) {
            d_key[atomicAdd(&d_bucket[range], 1)] = idx;
            __syncthreads();
        }
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 10; // Example size of key vector
    int range = 10; // Example range of values in key vector

    // Seed the random number generator
    std::srand(std::time(0));

    // Initialize and populate the key vector with random values
    std::vector<int> key(n);
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    // Allocate host memory for bucket
    std::vector<int> bucket(range + 1, 0); // Extra space for atomic counter

    // Allocate device memory
    int *d_key, *d_bucket;
    checkCudaError(cudaMalloc(&d_key, n * sizeof(int)), "Failed to allocate device memory for d_key");
    checkCudaError(cudaMalloc(&d_bucket, (range + 1) * sizeof(int)), "Failed to allocate device memory for d_bucket");

    // Copy key vector from host to device
    checkCudaError(cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy key data to device");

    // Initialize bucket vector on device
    checkCudaError(cudaMemset(d_bucket, 0, (range + 1) * sizeof(int)), "Failed to initialize device bucket memory");

    // Launch kernel to count occurrences
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    count_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_key, d_bucket, n);
    checkCudaError(cudaGetLastError(), "Count kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Count kernel execution failed");

    // Launch kernel to sort keys
    blocksPerGrid = (range + threadsPerBlock - 1) / threadsPerBlock;
    sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_key, d_bucket, n, range);
    checkCudaError(cudaGetLastError(), "Sort kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Sort kernel execution failed");

    // Copy sorted key vector from device to host
    checkCudaError(cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy sorted key data to host");

    // Free device memory
    cudaFree(d_key);
    cudaFree(d_bucket);

    // Print the sorted key vector
    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    return 0;
}
