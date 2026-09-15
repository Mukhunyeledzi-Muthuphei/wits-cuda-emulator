// 10 — BUG (caught before running): calling CPU-only code from a kernel.
//
// nvcc refuses to compile this. wcu does too, and explains why.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

float clampValue(float v)            // <-- a host function (no __device__)
{
    return v < 0 ? 0 : v;
}

__global__ void relu(float *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float *tmp = (float *)malloc(sizeof(float));   // <-- malloc on the GPU?
        x[i] = clampValue(x[i]);
        free(tmp);
    }
}

int main(void)
{
    int i = threadIdx.x;             // <-- threadIdx only exists inside kernels
    float *d_x;
    cudaMalloc((void **)&d_x, 16 * sizeof(float));
    relu(d_x, 16);                   // <-- kernels need <<< >>>
    cudaFree(d_x);
    return i;
}
