#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

using namespace std;
__device__ int nx = 41;
__device__ int ny = 41;
__device__ int nt = 500;
__device__ int nit = 50;
__device__ double dt = .01;
__device__ double rho = 1.;
__device__ double nu = .02;
__device__ double dxy= dx * dy;




__global__ void compute_bij(float *b, float *v, float *u, double xy_bytes, int dx, int dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num = i+(nx-1);
    if(num < (blockIdx.x-1)*(threadIdx.x -1))
    b[num] = rho * (1/dt * \
    (u[num+1] - u[num-1]) / (2 * dx) + (v[num+(nx-1)] - v[num-(nx -1)]) / (2 *dy) - \
    powf(((u[num+1]-u[num-1]) / (2*dx)), 2) - 2* ((u[num +(nx-1)] - u[num-nx +1]) / (2 * dy) * \
    (v[num+1] - v[num-1])/ (2* dx)) - powf(((v[num+nx-1] - v[num-nx+1])/(2*dy)), 2));
}


int main(){
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. /(nx - 1);
    double dy = 2. /(ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;
    double dxy= dx * dy;
    double xy_bytes = dxy * (sizeof(float ));
    printf("Matrix size: nx %d ny %d\n", nx, ny );

    float *u, *v, *p, *b, *un, *vn, *pn;
    cudaMallocManaged(&u,xy_bytes);
    cudaMallocManaged(&v,xy_bytes);
    cudaMallocManaged(&p,xy_bytes);
    cudaMallocManaged(&un,xy_bytes);
    cudaMallocManaged(&vn,xy_bytes);
    cudaMallocManaged(&pn,xy_bytes);
    cudaMallocManaged(&b,xy_bytes);

    for (int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            u[ny*i+j]= 0;
    for (int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            v[ny*i+j]= 0;
    for (int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            p[ny*i+j]= 0;
    for (int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            un[ny*i+j]= 0;
    for (int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            b[ny*i+j]= 0;
//    for (int i=0; i<nx; i++)
//        for(int j=0; j<ny; j++)
//            pn[ny*i+j]= 0;

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");
    for(int n = 0; n< nt; n++){
        compute_bij<<<nx * ny-1, ny>>>(b, v, u, xy_bytes, dx, dy);
    }
    for (int it = 0; it < nit; it ++){

    }
}
