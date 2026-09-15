/*
 * wcu.h — a teaching emulator for CUDA C.
 *
 * Programs written in (C-style) CUDA are translated by `wcu-translate`
 * into plain C that calls into this runtime. Kernels run on the CPU, one
 * GPU thread at a time, while the runtime checks every device-memory access
 * and records a trace for the visualizer.
 *
 * Nothing here is fast. Everything here is meant to be explainable.
 */
#ifndef WCU_H
#define WCU_H

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

cudaError wcu_malloc(void **devPtr, size_t size, const char *expr, const char *file, int line);
cudaError wcu_free(void *devPtr, const char *expr, const char *file, int line);
cudaError wcu_malloc_host(void **ptr, size_t size, const char *expr, const char *file, int line);
cudaError wcu_free_host(void *ptr, const char *file, int line);
cudaError wcu_memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                       const char *dst_expr, const char *src_expr, const char *file, int line);
cudaError wcu_memset(void *devPtr, int value, size_t count, const char *expr, const char *file, int line);
cudaError wcu_synchronize(const char *file, int line);
cudaError wcu_get_last_error(int peek, const char *file, int line);
cudaError wcu_mem_get_info(size_t *free_bytes, size_t *total_bytes);
cudaError wcu_device_reset(const char *file, int line);

const char *cudaGetErrorName(cudaError error);
const char *cudaGetErrorString(cudaError error);
cudaError cudaGetDeviceCount(int *count);
cudaError cudaSetDevice(int device);
cudaError cudaGetDevice(int *device);

#define cudaMalloc(devPtr, size)       wcu_malloc((void **)(devPtr), (size), #devPtr, __FILE__, __LINE__)
#define cudaFree(devPtr)               wcu_free((devPtr), #devPtr, __FILE__, __LINE__)
#define cudaMallocHost(ptr, size)      wcu_malloc_host((void **)(ptr), (size), #ptr, __FILE__, __LINE__)
#define cudaFreeHost(ptr)              wcu_free_host((ptr), __FILE__, __LINE__)
#define cudaMemcpy(dst, src, n, kind)  wcu_memcpy((dst), (src), (n), (kind), #dst, #src, __FILE__, __LINE__)
#define cudaMemset(devPtr, value, n)   wcu_memset((devPtr), (value), (n), #devPtr, __FILE__, __LINE__)
#define cudaDeviceSynchronize()        wcu_synchronize(__FILE__, __LINE__)
#define cudaGetLastError()             wcu_get_last_error(0, __FILE__, __LINE__)
#define cudaPeekAtLastError()          wcu_get_last_error(1, __FILE__, __LINE__)
#define cudaMemGetInfo(free_, total_)  wcu_mem_get_info((free_), (total_))
#define cudaDeviceReset()              wcu_device_reset(__FILE__, __LINE__)

/* ---- dim3 helpers (the translator rewrites dim3 b(16, 16) into these) --- */

static inline dim3 wcu_dim3_same(dim3 d) { return d; }
dim3 wcu_dim3_from_int(long long n);
dim3 wcu_dim3_n(const long long *v, size_t n);

#define WCU_TO_DIM3(e) _Generic((e), dim3: wcu_dim3_same, default: wcu_dim3_from_int)(e)
#define WCU_DIM3(...) \
    wcu_dim3_n((const long long[]){__VA_ARGS__}, sizeof((long long[]){__VA_ARGS__}) / sizeof(long long))

/* ---- Built-in kernel variables ------------------------------------------ */

typedef struct wcu_tls_state {
    dim3 thread_idx, block_idx, block_dim, grid_dim;
    int in_kernel;
    long long thread_id;   /* linear id of the GPU thread being run */
    void *kernel_fp;       /* frame address of the running kernel */
} wcu_tls_state;

extern _Thread_local wcu_tls_state wcu_tls;

static inline dim3 wcu_builtin_thread_idx(void) { return wcu_tls.thread_idx; }
static inline dim3 wcu_builtin_block_idx(void)  { return wcu_tls.block_idx; }
static inline dim3 wcu_builtin_block_dim(void)  { return wcu_tls.block_dim; }
static inline dim3 wcu_builtin_grid_dim(void)   { return wcu_tls.grid_dim; }

#define threadIdx wcu_builtin_thread_idx()
#define blockIdx  wcu_builtin_block_idx()
#define blockDim  wcu_builtin_block_dim()
#define gridDim   wcu_builtin_grid_dim()

/* ---- Kernel launches ---------------------------------------------------- */

typedef struct wcu_launch {
    int id;
    int ok;
    long long total, next;
} wcu_launch;

int  wcu_launch_begin(wcu_launch *L, const char *kernel, dim3 grid, dim3 block,
                        const char *file, int line);
int  wcu_launch_next(wcu_launch *L);
void wcu_launch_end(wcu_launch *L);
void wcu_kernel_enter(void *frame, const char *kernel, int first_line, int last_line);

/* ---- Checked memory access (inserted into kernel bodies) ---------------- */

enum { WCU_READ = 1, WCU_WRITE = 2, WCU_RW = 3, WCU_ADDR = 4, WCU_ATOMIC = 8 };

void *wcu_access(void *base, size_t elem_size, long long index, int rw, int type,
                   const char *name, const char *file, int line);

#define WCU_TYPECODE(x) _Generic((x),                                     \
    float: 1, double: 2, char: 3, signed char: 3, unsigned char: 4,         \
    short: 5, unsigned short: 6, int: 7, unsigned int: 8, long: 9,          \
    unsigned long: 10, long long: 11, unsigned long long: 12, _Bool: 13,    \
    default: 0)

#define WCU_AT(p, i, rw)                                                  \
    (*(__typeof__(&(p)[0]))wcu_access((void *)&(p)[0], sizeof((p)[0]),   \
        (long long)(i), (rw), WCU_TYPECODE((p)[0]), #p, __FILE__, __LINE__))

/* ---- Atomics -------------------------------------------------------------
 * Each one reads the old value, updates memory and returns the old value, as
 * one indivisible step. The runtime knows these accesses are atomic, so it
 * does not report them as data races.
 */
#define WCU_ATOMIC_PTR(p)                                                 \
    ((__typeof__(p))wcu_access((void *)(p), sizeof(*(p)), 0,             \
        WCU_RW | WCU_ATOMIC, WCU_TYPECODE(*(p)), #p, __FILE__, __LINE__))

#define WCU_ATOMIC_OP(p, expr)                                            \
    ({ __auto_type _wcu_p = WCU_ATOMIC_PTR(p);                          \
       __typeof__(*_wcu_p) _wcu_old = *_wcu_p, _wcu_v = (expr);     \
       *_wcu_p = _wcu_v; _wcu_old; })

#define atomicAdd(p, v)  WCU_ATOMIC_OP((p), _wcu_old + (v))
#define atomicSub(p, v)  WCU_ATOMIC_OP((p), _wcu_old - (v))
#define atomicExch(p, v) WCU_ATOMIC_OP((p), (v))
#define atomicMax(p, v)  WCU_ATOMIC_OP((p), _wcu_old > (v) ? _wcu_old : (v))
#define atomicMin(p, v)  WCU_ATOMIC_OP((p), _wcu_old < (v) ? _wcu_old : (v))
#define atomicAnd(p, v)  WCU_ATOMIC_OP((p), _wcu_old & (v))
#define atomicOr(p, v)   WCU_ATOMIC_OP((p), _wcu_old | (v))
#define atomicXor(p, v)  WCU_ATOMIC_OP((p), _wcu_old ^ (v))
#define atomicCAS(p, cmp, v) WCU_ATOMIC_OP((p), _wcu_old == (cmp) ? (v) : _wcu_old)

#ifdef __cplusplus
}
#endif

#endif /* WCU_H */
