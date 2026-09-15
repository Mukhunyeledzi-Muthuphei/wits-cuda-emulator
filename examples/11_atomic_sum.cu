// 11 — The fix for example 05: adding with an atomic.
//
// atomicAdd does read-modify-write as one indivisible step, so threads can
// safely update the same element. It is slower than giving each thread its
// own output, but it is correct.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024

__global__ void total(const int *a, int *sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&sum[0], a[i]);
}

int main(void)
{
    int a[N];
    for (int i = 0; i < N; i++) a[i] = 1;

    int *d_a, *d_sum;
    cudaMalloc((void **)&d_a, sizeof a);
    cudaMalloc((void **)&d_sum, sizeof(int));
    cudaMemcpy(d_a, a, sizeof a, cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(int));

    total<<<4, 256>>>(d_a, d_sum, N);

    int sum;
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum = %d (expected %d)\n", sum, N);

    cudaFree(d_a);
    cudaFree(d_sum);
    return 0;
}
