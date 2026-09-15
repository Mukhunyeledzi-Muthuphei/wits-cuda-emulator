// 07 — BUG: reading GPU memory directly from the CPU.
//
// d_c lives in GPU memory. The CPU cannot read it without cudaMemcpy, so
// this crashes on a real machine too (usually with a bare "Segmentation fault").
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 64

__global__ void fill(int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = i * i;
}

int main(void)
{
    int *d_c;
    cudaMalloc((void **)&d_c, N * sizeof(int));
    fill<<<1, 64>>>(d_c, N);
    cudaDeviceSynchronize();

    printf("c[7] = %d\n", d_c[7]);   // <-- d_c is a device pointer

    cudaFree(d_c);
    return 0;
}
