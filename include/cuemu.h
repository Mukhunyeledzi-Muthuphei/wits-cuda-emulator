/*
 * cuemu.h — a teaching emulator for CUDA C.
 *
 * Programs written in (C-style) CUDA are translated by `cuemu-translate`
 * into plain C that calls into this runtime. Kernels run on the CPU, one
 * GPU thread at a time, while the runtime checks every device-memory access
 * and records a trace for the visualizer.
 *
 * Nothing here is fast. Everything here is meant to be explainable.
 */
#ifndef CUEMU_H
#define CUEMU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Function qualifiers ------------------------------------------------ */

#define __global__
#define __device__
#define __host__

/* ---- Types -------------------------------------------------------------- */

typedef struct dim3 {
    unsigned int x, y, z;
} dim3;

typedef enum cudaError {
    cudaSuccess                   = 0,
    cudaErrorInvalidValue         = 1,
    cudaErrorMemoryAllocation     = 2,
    cudaErrorInitializationError  = 3,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidDevice        = 101,
    cudaErrorIllegalAddress       = 700,
} cudaError;
typedef cudaError cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} cudaMemcpyKind;

/* ---- Runtime entry points (use the CUDA names below instead) ----------- */

cudaError cuemu_malloc(void **devPtr, size_t size, const char *expr, const char *file, int line);
cudaError cuemu_free(void *devPtr, const char *expr, const char *file, int line);
cudaError cuemu_malloc_host(void **ptr, size_t size, const char *expr, const char *file, int line);
cudaError cuemu_free_host(void *ptr, const char *file, int line);
cudaError cuemu_memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                       const char *dst_expr, const char *src_expr, const char *file, int line);
cudaError cuemu_memset(void *devPtr, int value, size_t count, const char *expr, const char *file, int line);
cudaError cuemu_synchronize(const char *file, int line);
cudaError cuemu_get_last_error(int peek, const char *file, int line);
cudaError cuemu_mem_get_info(size_t *free_bytes, size_t *total_bytes);
cudaError cuemu_device_reset(const char *file, int line);

const char *cudaGetErrorName(cudaError error);
const char *cudaGetErrorString(cudaError error);
cudaError cudaGetDeviceCount(int *count);
cudaError cudaSetDevice(int device);
cudaError cudaGetDevice(int *device);

#define cudaMalloc(devPtr, size)       cuemu_malloc((void **)(devPtr), (size), #devPtr, __FILE__, __LINE__)
#define cudaFree(devPtr)               cuemu_free((devPtr), #devPtr, __FILE__, __LINE__)
#define cudaMallocHost(ptr, size)      cuemu_malloc_host((void **)(ptr), (size), #ptr, __FILE__, __LINE__)
#define cudaFreeHost(ptr)              cuemu_free_host((ptr), __FILE__, __LINE__)
#define cudaMemcpy(dst, src, n, kind)  cuemu_memcpy((dst), (src), (n), (kind), #dst, #src, __FILE__, __LINE__)
#define cudaMemset(devPtr, value, n)   cuemu_memset((devPtr), (value), (n), #devPtr, __FILE__, __LINE__)
#define cudaDeviceSynchronize()        cuemu_synchronize(__FILE__, __LINE__)
#define cudaGetLastError()             cuemu_get_last_error(0, __FILE__, __LINE__)
#define cudaPeekAtLastError()          cuemu_get_last_error(1, __FILE__, __LINE__)
#define cudaMemGetInfo(free_, total_)  cuemu_mem_get_info((free_), (total_))
#define cudaDeviceReset()              cuemu_device_reset(__FILE__, __LINE__)

/* ---- dim3 helpers (the translator rewrites dim3 b(16, 16) into these) --- */

static inline dim3 cuemu_dim3_same(dim3 d) { return d; }
dim3 cuemu_dim3_from_int(long long n);
dim3 cuemu_dim3_n(const long long *v, size_t n);

#define CUEMU_TO_DIM3(e) _Generic((e), dim3: cuemu_dim3_same, default: cuemu_dim3_from_int)(e)
#define CUEMU_DIM3(...) \
    cuemu_dim3_n((const long long[]){__VA_ARGS__}, sizeof((long long[]){__VA_ARGS__}) / sizeof(long long))

/* ---- Built-in kernel variables ------------------------------------------ */

typedef struct cuemu_tls_state {
    dim3 thread_idx, block_idx, block_dim, grid_dim;
    int in_kernel;
    long long thread_id;   /* linear id of the GPU thread being run */
    void *kernel_fp;       /* frame address of the running kernel */
} cuemu_tls_state;

extern _Thread_local cuemu_tls_state cuemu_tls;

static inline dim3 cuemu_builtin_thread_idx(void) { return cuemu_tls.thread_idx; }
static inline dim3 cuemu_builtin_block_idx(void)  { return cuemu_tls.block_idx; }
static inline dim3 cuemu_builtin_block_dim(void)  { return cuemu_tls.block_dim; }
static inline dim3 cuemu_builtin_grid_dim(void)   { return cuemu_tls.grid_dim; }

#define threadIdx cuemu_builtin_thread_idx()
#define blockIdx  cuemu_builtin_block_idx()
#define blockDim  cuemu_builtin_block_dim()
#define gridDim   cuemu_builtin_grid_dim()

/* ---- Kernel launches ---------------------------------------------------- */

typedef struct cuemu_launch {
    int id;
    int ok;
    long long total, next;
} cuemu_launch;

int  cuemu_launch_begin(cuemu_launch *L, const char *kernel, dim3 grid, dim3 block,
                        const char *file, int line);
int  cuemu_launch_next(cuemu_launch *L);
void cuemu_launch_end(cuemu_launch *L);
void cuemu_kernel_enter(void *frame, const char *kernel, int first_line, int last_line);

/* ---- Checked memory access (inserted into kernel bodies) ---------------- */

enum { CUEMU_READ = 1, CUEMU_WRITE = 2, CUEMU_RW = 3, CUEMU_ADDR = 4, CUEMU_ATOMIC = 8 };

void *cuemu_access(void *base, size_t elem_size, long long index, int rw, int type,
                   const char *name, const char *file, int line);

#define CUEMU_TYPECODE(x) _Generic((x),                                     \
    float: 1, double: 2, char: 3, signed char: 3, unsigned char: 4,         \
    short: 5, unsigned short: 6, int: 7, unsigned int: 8, long: 9,          \
    unsigned long: 10, long long: 11, unsigned long long: 12, _Bool: 13,    \
    default: 0)

#define CUEMU_AT(p, i, rw)                                                  \
    (*(__typeof__(&(p)[0]))cuemu_access((void *)&(p)[0], sizeof((p)[0]),   \
        (long long)(i), (rw), CUEMU_TYPECODE((p)[0]), #p, __FILE__, __LINE__))

/* ---- Atomics -------------------------------------------------------------
 * Each one reads the old value, updates memory and returns the old value, as
 * one indivisible step. The runtime knows these accesses are atomic, so it
 * does not report them as data races.
 */
#define CUEMU_ATOMIC_PTR(p)                                                 \
    ((__typeof__(p))cuemu_access((void *)(p), sizeof(*(p)), 0,             \
        CUEMU_RW | CUEMU_ATOMIC, CUEMU_TYPECODE(*(p)), #p, __FILE__, __LINE__))

#define CUEMU_ATOMIC_OP(p, expr)                                            \
    ({ __auto_type _cuemu_p = CUEMU_ATOMIC_PTR(p);                          \
       __typeof__(*_cuemu_p) _cuemu_old = *_cuemu_p, _cuemu_v = (expr);     \
       *_cuemu_p = _cuemu_v; _cuemu_old; })

#define atomicAdd(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old + (v))
#define atomicSub(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old - (v))
#define atomicExch(p, v) CUEMU_ATOMIC_OP((p), (v))
#define atomicMax(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old > (v) ? _cuemu_old : (v))
#define atomicMin(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old < (v) ? _cuemu_old : (v))
#define atomicAnd(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old & (v))
#define atomicOr(p, v)   CUEMU_ATOMIC_OP((p), _cuemu_old | (v))
#define atomicXor(p, v)  CUEMU_ATOMIC_OP((p), _cuemu_old ^ (v))
#define atomicCAS(p, cmp, v) CUEMU_ATOMIC_OP((p), _cuemu_old == (cmp) ? (v) : _cuemu_old)

#ifdef __cplusplus
}
#endif

#endif /* CUEMU_H */
