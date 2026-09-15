// Valid code the translator must not break.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

typedef struct { float x, y; } point_t;
typedef struct Particle { float pos; float vel; } Particle;

#define IDX(x, y, w) ((y) * (w) + (x))
#define W 13
#define H 7

__host__ __device__ static inline float sq(float v) { return v * v; }

__device__ float dot(const float *a, const float *b, int n)
{
    float s = 0, tmp[4] = {0, 0, 0, 0};
    int k, arr[3], *q = arr;
    for (k = 0; k < n; k++) s += a[k] * b[k];
    tmp[0] = s;
    arr[0] = 1;
    q[1] = 2;
    *q = 3;
    return tmp[0] + (float)(arr[0] - 3) + (float)(q[1] - 2);
}

__global__ void edge(float *out, const float *m, Particle *ps, point_t *pts, int *ids, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int p = IDX(x, y, w);
    float local[3];
    local[0] = m[p];
    local[1] = local[0] * 2;
    local[2] = sq(local[1]);
    float *mp = &out[p];
    *mp = local[2] + dot(&m[p], &m[p], 1);
    ps[p].pos += ps[p].vel;
    Particle *pp = &ps[p];
    pp->vel = pp->vel * 0.5f;
    pts[p].x = (float)x;
    pts[p].y = (float)y;
    *(ids + p) = p;
    out[m[p] > 1e9f ? 0 : p] += 0;
}

__global__ void cube(int *v, int n)
{
    int i = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    i = ((i * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (i < n)
        v[i]++;
}

int main(void)
{
    int n = W * H;
    float *m = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) m[i] = (float)i;
    Particle *ps = (Particle *)calloc(n, sizeof(Particle));
    for (int i = 0; i < n; i++) ps[i].vel = 2;

    float *d_out, *d_m;
    Particle *d_ps;
    point_t *d_pts;
    int *d_ids;
    cudaMalloc((void **)&d_out, n * sizeof(float));
    cudaMalloc((void **)&d_m, n * sizeof(float));
    cudaMalloc((void **)&d_ps, n * sizeof(Particle));
    cudaMalloc((void **)&d_pts, n * sizeof(point_t));
    cudaMalloc((void **)&d_ids, n * sizeof(int));
    cudaMemcpy(d_m, m, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ps, ps, n * sizeof(Particle), cudaMemcpyHostToDevice);

    dim3 b(4, 4), unused;
    dim3 g = {(W + 3) / 4, (H + 3) / 4, 1};
    edge<<<g, b>>>(d_out, d_m + 0,   // a comment inside the launch
                   d_ps, d_pts,
                   d_ids, W, H);
    if (cudaGetLastError() != cudaSuccess) { printf("launch failed\n"); return 1; }

    float *out = (float *)malloc(n * sizeof(float));
    point_t *pts = (point_t *)malloc(n * sizeof(point_t));
    int *ids = (int *)malloc(n * sizeof(int));
    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ps, d_ps, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(pts, d_pts, n * sizeof(point_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids, d_ids, n * sizeof(int), cudaMemcpyDeviceToHost);

    int ok = unused.x == 1 && unused.z == 1;
    for (int i = 0; i < n; i++) {
        float want = (2 * m[i]) * (2 * m[i]) + m[i] * m[i];
        if (out[i] != want || ps[i].pos != 2 || ps[i].vel != 1 || ids[i] != i ||
            pts[i].x != (float)(i % W) || pts[i].y != (float)(i / W))
            ok = 0;
    }

    int cn = 2 * 3 * 2 * 2 * 2 * 2;
    int *v = (int *)calloc(cn, sizeof(int)), *d_v;
    cudaMalloc((void **)&d_v, cn * sizeof(int));
    cudaMemcpy(d_v, v, cn * sizeof(int), cudaMemcpyHostToDevice);
    cube<<<dim3(2, 3, 2), dim3(2, 2, 2)>>>(d_v, cn);
    cudaMemcpy(v, d_v, cn * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < cn; i++)
        if (v[i] != 1) ok = 0;

    printf("%s\n", ok ? "EDGE OK" : "EDGE FAILED");
    cudaFree(d_out); cudaFree(d_m); cudaFree(d_ps); cudaFree(d_pts); cudaFree(d_ids); cudaFree(d_v);
    free(m); free(ps); free(out); free(pts); free(ids); free(v);
    return ok ? 0 : 1;
}
