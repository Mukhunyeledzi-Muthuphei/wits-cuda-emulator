// 01 — Vector addition: the "hello world" of CUDA. This program is correct.
//
//   cuemu run --open examples/01_vector_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000

// A kernel runs once per GPU thread. Each thread works out which element is
// "its" element from its block and thread index.
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    size_t bytes = N * sizeof(float);

    // 1. Host (CPU) memory
    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
    }

    // 2. Device (GPU) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    // 3. Copy the inputs to the GPU
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // 4. Launch: round the number of blocks up so every element gets a thread
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "kernel launch failed\n");
        return 1;
    }

    // 5. Copy the result back (this waits for the kernel to finish)
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (c[i] != a[i] + b[i]) ok = 0;
    printf("c[0] = %.1f, c[%d] = %.1f -> %s\n", c[0], N - 1, c[N - 1], ok ? "correct" : "WRONG");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
