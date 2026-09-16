# WCU — the Wits CUDA emulator

WCU (command: `wcu`) runs CUDA C programs on the CPU. It is not fast, and it is not meant to be: it exists so
that the ideas of CUDA — blocks and threads, separate host and device memory, copies across a
slow link, races between threads that run at the same time — can be learned, seen and debugged
on any Mac or Linux machine.

It does three things a real GPU does not:

1. **Explains mistakes.** Out-of-bounds writes, host pointers passed to kernels, reads of memory
   nobody wrote, races, block counts rounded down — each is reported with the source line, the
   GPU thread that did it, the array involved and what to do about it.
2. **Draws the run.** Every program writes a self-contained HTML page: a timeline, the grid of
   blocks and threads, which thread touched which element of which array, with the values.
3. **Simulates GPU time.** Real timings on your CPU would teach the wrong lessons, so wcu
   models the costs a GPU actually has (start-up, copies over PCIe, parallel execution) and
   reports those instead.

Code written for WCU is ordinary CUDA C: `__global__`, `<<<blocks, threads>>>`, `cudaMalloc`,
`cudaMemcpy`, `threadIdx`. It is meant to move to real hardware unchanged.

## Getting started

Requirements: macOS or Linux, `clang` and `make` (on a Mac, `xcode-select --install`).

```sh
make                                          # builds the translator and runtime
bin/wcu run --open examples/01_vector_add.cu
```

`--open` opens the visualization in your browser; without it, the path is printed at the end.
The page has a sidebar with the run status, the timeline, the list of steps and the problems
found; the ◀ ▶ buttons above the detail panel step through the run.
Add `bin` to your `PATH` to use `wcu` (and the `nvcc` stand-in) from anywhere.

```
wcu run [--open] file.cu [-- program arguments]
wcu build file.cu [more files] [-o program] [compiler flags]
wcu translate file.cu            # show the plain C that wcu generates
```

Environment variables:

| Variable | Meaning |
| --- | --- |
| `WCU_ORDER=sequential` | run GPU threads in order (default: shuffled, like real hardware) |
| `WCU_STRICT=1` | stop at the first error |
| `WCU_DEVICE_MB=512` | size of the simulated device memory (default 4096) |
| `WCU_SEED=n` | change the thread shuffle |
| `WCU_TRACE=0` | skip writing the visualization |
| `WCU_TRACE_OUT=path` | write the visualization somewhere else (`.json` writes raw trace data) |

## The examples

Each one is a small, complete program. The numbered bugs are the mistakes students actually make.

| File | What it shows |
| --- | --- |
| `01_vector_add.cu` | a correct program, end to end |
| `02_missing_bounds_check.cu` | no `if (i < n)`: 1024 threads, 1000 elements |
| `03_host_pointer_in_kernel.cu` | passing `a` (host) instead of `d_a` (device) |
| `04_forgot_to_copy.cu` | a kernel reading device memory nobody filled |
| `05_race_condition.cu` | every thread updating `sum[0]` |
| `06_grayscale_2d.cu` | a correct 2D grid of 2D blocks |
| `07_device_pointer_on_host.cu` | the CPU reading GPU memory directly |
| `08_block_count_rounding.cu` | `N / threadsPerBlock` silently dropping the tail |
| `09_bytes_not_elements.cu` | `cudaMemcpy` counting bytes, not elements |
| `10_host_function_in_kernel.cu` | mistakes caught before the program runs |
| `11_atomic_sum.cu` | the fix for 05, using `atomicAdd` |
| `12_coalescing.cu` | warps: the same work, done twice, with 32× the memory traffic |

## What WCU checks

Reported while the program runs, with source line and thread:

`out-of-bounds-write` · `out-of-bounds-read` · `uninitialized-read` · `data-race` ·
`host-memory-in-kernel` · `device-pointer-on-host` · `use-after-free` · `null-pointer` ·
`partial-coverage` (a launch that computed only part of an array) · `invalid-configuration` ·
`memcpy-swapped` · `memcpy-wrong-side` · `memcpy-overflow` · `memcpy-partial` ·
`memcpy-uninitialized` · `double-free` · `free-host-pointer` · `out-of-memory` ·
`device-error-state` · `memory-leak` · `unchecked-error` · `uncoalesced-access`

Caught at compile time, the way `nvcc` would: calling a host function from a kernel, calling a
kernel without `<<< >>>`, using `threadIdx` in host code, a kernel returning something other
than `void`, a kernel launching a kernel.

A program with errors exits with status 1, so assignments can be graded by running them.

## How it works

```
your.cu ──wcu-translate──► plain C ──clang──► program ──links──► libwcu.a
```

- **`src/translate.c`** rewrites CUDA syntax into C: `<<< >>>` becomes a loop over GPU threads,
  and every `a[i]`, `*p` and `p->x` inside `__global__`/`__device__` code becomes a checked
  access. Line numbers are preserved, so compiler errors point into your `.cu` file.
- **`src/runtime.c`** is the emulated GPU. Device memory lives at addresses inside a reserved,
  unreadable region, so the CPU touching a device pointer faults and wcu can explain it; the
  real bytes sit in ordinary buffers that only kernels reach, through the checked accesses. The
  runtime tracks, per byte, whether anything ever wrote it, and, per element per launch, which
  thread read or wrote it — that is where race and uninitialized-read detection come from.
- **`viewer/viewer.html`** is the visualization; the runtime embeds the trace into a copy of it.

Threads run one at a time, but grouped into **warps** of 32: the lanes of a warp run back to
back, and the warps themselves are shuffled. While a warp runs, WCU records which 128-byte
transactions its lanes touched, which is how it measures coalescing — one transaction shared by
32 neighbouring threads, or 32 separate ones for a scattered warp. The shuffled warp order is
deliberately unstable: code whose result depends on it is buggy on real hardware, and shuffling
helps expose it (`printf` from a kernel comes out jumbled, as it does on a GPU).

### The simulated clock

Times in the summary and the visualization come from this model — a teaching device, not a
measurement of your machine:

| Quantity | Value |
| --- | --- |
| CUDA start-up (first call) | 80 ms |
| kernel launch overhead | 0.02 ms |
| warp size | 32 threads |
| warps resident at once | 64 (2048 threads) |
| memory transaction | 128 bytes, 0.25 ns each device-wide |
| instruction issued to one warp | 20 ns |
| memory access, one CPU core | 0.4 ns |
| host ⇄ device copy | 6 GB/s (12 GB/s from `cudaMallocHost`) |

A kernel's time is `launch overhead + max(memory, issue)`: the transactions its warps actually
needed against the instructions its warps had to issue, since a GPU overlaps the two. That is
why scattering an access pattern makes the modelled kernel slower — it really does need more
transactions.

Host code is timed for real, since it runs natively. Launches are asynchronous, as on a GPU: the
CPU continues, and a copy or `cudaDeviceSynchronize` waits for the device. The lesson that
survives the move to real hardware is the shape of the bars, not their exact length: copies and
start-up dominate small problems, and kernels only pay off when there is enough work.

## Limits

Supported: `__global__`, `__device__`, `__host__`, `dim3`, 1D/2D/3D grids and blocks,
`threadIdx`/`blockIdx`/`blockDim`/`gridDim`, `cudaMalloc`, `cudaFree`, `cudaMallocHost`,
`cudaFreeHost`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`,
`cudaGetErrorString`/`Name`, `cudaMemGetInfo`, `cudaDeviceReset`, the `atomic*` functions,
`printf` in kernels, and math functions.

Not supported yet:

- **`__shared__` memory and `__syncthreads()`** — these need threads in a block to pause at a
  barrier, which the one-thread-at-a-time design cannot do. This is the largest gap; it rules
  out tiled matrix multiply and block-level reductions.
- **Branch divergence.** Warps are modelled for memory (coalescing) but not for control flow:
  `if (threadIdx.x % 2)` costs a real warp both sides of the branch, and costs nothing here.
- Streams, events, unified memory, warp intrinsics (`__shfl_sync`, `__ballot_sync`), textures,
  dynamic parallelism.
- Libraries: cuBLAS, cuRAND, Thrust.
- C++ in `.cu` files (templates, classes, `new`). Write C, as `nvcc` also accepts it.
- Inside kernels, memory reached in ways the translator cannot see (a pointer cast in a macro,
  `memcpy`) is not checked; the program stops with a message saying so.

## For instructors

- `make test` runs every example and test case against `tests/expected.txt`, and renders the
  visualizer for each trace under a DOM stub (needs `node`; skipped without it).
- `tests/cases/ok_edge_cases.cu` is the regression test for the translator: valid CUDA that must
  keep working (structs, `->`, local arrays, `__device__` helpers, 3D launches, macros).
- To add a new check, add a diagnostic in `src/runtime.c` (see `report()` and `flush_kernel_diags()`)
  and an expectation in `tests/expected.txt`.
- Student work can be graded by exit status, and the `.wcu.html` page attached to feedback:
  it is one self-contained file that opens offline.
