// 02 — BUG: no bounds check.
//
// 4 blocks x 256 threads = 1024 GPU threads, but the arrays only have 1000
// elements. On a real GPU the extra 24 threads silently scribble over memory.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000

__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];          // <-- what about i >= n?
}

int main(void)
{
    size_t bytes = N * sizeof(float);
    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("c[999] = %.1f\n", c[999]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
