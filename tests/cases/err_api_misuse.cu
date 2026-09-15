// API mistakes the runtime should explain.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void touch(int *v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = i;
}

int main(void)
{
    int host[16];
    int *d_v;
    cudaMalloc((void **)&d_v, sizeof host);

    dim3 threads = {4, 4};                              // z = 0
    touch<<<1, threads>>>(d_v, 16);

    cudaMemcpy(host, d_v, sizeof host, cudaMemcpyHostToDevice);   // swapped

    int *d_w;
    cudaMalloc((void **)&d_w, sizeof host);
    cudaFree(d_w);
    touch<<<1, 16>>>(d_w, 16);                          // use after free
    cudaFree(d_w);                                      // double free

    cudaFree(d_v);
    return 0;
}
