// 05 — BUG: a data race.
//
// Every GPU thread does `sum[0] += a[i]` at the same time. On a real GPU most
// of those updates are lost and the total is wrong (and different every run).
// cuemu runs threads one at a time, so the answer happens to look right here,
// which is exactly why it reports the race.
//
// The fix is in examples/11_atomic_sum.cu.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024

__global__ void total(const int *a, int *sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        sum[0] += a[i];          // <-- every thread writes the same element
}

int main(void)
{
    int a[N];
    for (int i = 0; i < N; i++) a[i] = 1;

    int *d_a, *d_sum;
    cudaMalloc((void **)&d_a, sizeof a);
    cudaMalloc((void **)&d_sum, sizeof(int));
    cudaMemcpy(d_a, a, sizeof a, cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_sum, &zero, sizeof(int), cudaMemcpyHostToDevice);

    total<<<4, 256>>>(d_a, d_sum, N);

    int sum;
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum = %d (expected %d)\n", sum, N);

    cudaFree(d_a);
    cudaFree(d_sum);
    return 0;
}
