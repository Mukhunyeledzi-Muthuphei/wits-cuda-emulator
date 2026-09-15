// 03 — BUG: passing host memory to a kernel.
//
// The kernel receives `a` (from malloc) instead of `d_a` (from cudaMalloc).
// GPU threads cannot see CPU memory, so a real GPU reports
// "an illegal memory access was encountered".
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 512

__global__ void scale(float *x, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] * factor;
}

int main(void)
{
    size_t bytes = N * sizeof(float);
    float *a = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) a[i] = (float)i;

    float *d_a;
    cudaMalloc((void **)&d_a, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    scale<<<2, 256>>>(a, 2.0f, N);   // <-- should be d_a

    cudaError err = cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
    printf("copy back: %s\n", cudaGetErrorName(err));

    cudaFree(d_a);
    free(a);
    return 0;
}
