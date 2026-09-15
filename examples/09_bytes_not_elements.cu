// 09 — BUG: passing the number of elements where cudaMemcpy wants bytes.
//
// cudaMemcpy(d_a, a, N, ...) copies N *bytes*: a quarter of the float array.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 128

__global__ void negate(float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = -a[i];
}

int main(void)
{
    float *a = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) a[i] = (float)i;

    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMemcpy(d_a, a, N, cudaMemcpyHostToDevice);        // <-- N bytes, not N floats

    negate<<<1, N>>>(d_a, N);

    cudaMemcpy(a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("a[100] = %g (expected -100)\n", a[100]);

    cudaFree(d_a);
    free(a);
    return 0;
}
