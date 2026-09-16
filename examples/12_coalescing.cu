// 12 — Warps and coalescing: why the *order* of your indexes matters.
//
// Both kernels do exactly the same work: double every element of an array.
// The only difference is which element each thread picks up.
//
// A warp is 32 threads that issue their instructions together. When a warp
// reads memory, the hardware fetches whole 128-byte transactions. If the 32
// threads want 32 neighbouring floats, that is one transaction for the warp.
// If they want 32 floats scattered far apart, that is 32 transactions —
// the same work, but many times the memory traffic.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (1 << 20)          // 1,048,576 floats
#define LANES 32

// thread i takes element i: neighbours in the warp read neighbours in memory
__global__ void coalesced(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 2.0f * in[i];
}

// same elements, same count, but the warp's 32 lanes land far apart
__global__ void scattered(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i % LANES) * (n / LANES) + (i / LANES);
        out[j] = 2.0f * in[j];
    }
}

int main(void)
{
    size_t bytes = N * sizeof(float);
    float *a = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) a[i] = (float)i;

    float *d_a, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    coalesced<<<numBlocks, threadsPerBlock>>>(d_a, d_c, N);
    scattered<<<numBlocks, threadsPerBlock>>>(d_a, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("c[4242] = %.1f (expected %.1f)\n", c[4242], 2.0f * a[4242]);
    printf("Compare the two launches in the visualization: same work, different memory traffic.\n");

    cudaFree(d_a);
    cudaFree(d_c);
    free(a);
    free(c);
    return 0;
}
