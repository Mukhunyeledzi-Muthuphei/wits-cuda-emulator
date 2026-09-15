// 04 — BUG: forgetting to copy an input to the device.
//
// d_b is allocated but never filled, so the kernel reads whatever garbage
// was left in GPU memory.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 256

__global__ void saxpy(float alpha, const float *x, const float *y, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = alpha * x[i] + y[i];
}

int main(void)
{
    size_t bytes = N * sizeof(float);
    float *x = (float *)malloc(bytes);
    float *y = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 10.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc((void **)&d_x, bytes);
    cudaMalloc((void **)&d_y, bytes);
    cudaMalloc((void **)&d_out, bytes);
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    // <-- missing: cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

    saxpy<<<1, 256>>>(2.0f, d_x, d_y, d_out, N);
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("out[0] = %g (expected 12)\n", out[0]);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    free(x);
    free(y);
    free(out);
    return 0;
}
