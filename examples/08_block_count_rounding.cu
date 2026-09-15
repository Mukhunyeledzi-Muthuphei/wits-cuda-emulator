// 08 — BUG: rounding the block count down.
//
// N / threadsPerBlock = 1000 / 256 = 3 blocks = 768 threads, so the last 232
// elements are never computed. Nothing crashes; the answer is just wrong.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000

__global__ void square(const double *in, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i] * in[i];
}

int main(void)
{
    size_t bytes = N * sizeof(double);
    double *in = (double *)malloc(bytes);
    double *out = (double *)malloc(bytes);
    for (int i = 0; i < N; i++) in[i] = i;

    double *d_in, *d_out;
    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = N / threadsPerBlock;     // <-- rounds down
    square<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("out[999] = %g (expected 998001)\n", out[999]);

    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out);
    return 0;
}
