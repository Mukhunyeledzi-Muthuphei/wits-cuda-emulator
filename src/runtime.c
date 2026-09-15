/*
 * runtime.c — the cuemu runtime.
 *
 * Device memory lives at fake addresses inside a reserved, inaccessible
 * address range, so host code that dereferences a device pointer crashes
 * (and we explain why). The real bytes live in ordinary heap buffers that
 * kernels reach through cuemu_access(), which checks bounds, tracks which
 * bytes were ever written, detects data races between GPU threads and
 * records a trace for the visualizer.
 *
 * Timing is simulated with a deliberately simple model (see MODEL_*): the
 * point is to show where time goes on a real GPU, not to measure your Mac.
 */
#define _DARWIN_C_SOURCE 1
#include "cuemu.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/ucontext.h>
#endif

#ifndef CUEMU_DEFAULT_VIEWER
#define CUEMU_DEFAULT_VIEWER ""
#endif

/* ---- The simulated GPU ---------------------------------------------------- */

#define MODEL_CORES              2048    /* GPU threads that run at once       */
#define MODEL_GPU_NS_PER_OP      12.0    /* cost of one memory access on GPU   */
#define MODEL_CPU_NS_PER_OP      0.4     /* same work on one CPU core          */
#define MODEL_LAUNCH_MS          0.02    /* kernel launch overhead            */
#define MODEL_API_MS             0.005   /* host-side cost of an async call    */
#define MODEL_MALLOC_MS          0.05
#define MODEL_CONTEXT_INIT_MS    80.0    /* the first CUDA call is slow        */
#define MODEL_PCIE_PAGEABLE_GBPS 6.0
#define MODEL_PCIE_PINNED_GBPS   12.0
#define MODEL_PCIE_LATENCY_MS    0.01
#define MAX_THREADS_PER_BLOCK    1024

/* ---- Trace limits --------------------------------------------------------- */

#define MAX_EVENTS            4000
#define MAX_DIAG_JSON         400
#define MAX_DIAG_PRINTED      40
#define REC_MAX_THREAD_ID     65536      /* record accesses of threads below  */
#define REC_MAX_TOTAL         400000
#define REC_MAX_PER_THREAD    32
#define SNAP_MAX_BYTES        16384
#define SNAP_BUDGET           (48u << 20)
#define MAX_KDIAG             64

_Thread_local cuemu_tls_state cuemu_tls;

/* ---- String buffers ------------------------------------------------------- */

typedef struct { char *p; size_t n, cap; } sbuf;

static void sb_reserve(sbuf *b, size_t extra)
{
    if (b->n + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->n + extra + 1) cap *= 2;
    char *p = realloc(b->p, cap);
    if (!p) { fputs("cuemu: out of memory\n", stderr); abort(); }
    b->p = p;
    b->cap = cap;
}

static void sb_putn(sbuf *b, const char *s, size_t n)
{
    sb_reserve(b, n);
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void sb_puts(sbuf *b, const char *s) { sb_putn(b, s, strlen(s)); }

static void sb_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void sb_printf(sbuf *b, const char *fmt, ...)
{
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n > 0) {
        sb_reserve(b, (size_t)n);
        vsnprintf(b->p + b->n, (size_t)n + 1, fmt, ap2);
        b->n += (size_t)n;
    }
    va_end(ap2);
}

static void sb_json_str(sbuf *b, const char *s)
{
    sb_putn(b, "\"", 1);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  sb_puts(b, "\\\""); break;
        case '\\': sb_puts(b, "\\\\"); break;
        case '\n': sb_puts(b, "\\n"); break;
        case '\r': sb_puts(b, "\\r"); break;
        case '\t': sb_puts(b, "\\t"); break;
        case '<':  sb_puts(b, "\\u003c"); break;
        default:
            if (c < 0x20) sb_printf(b, "\\u%04x", c);
            else sb_putn(b, (const char *)&c, 1);
        }
    }
    sb_putn(b, "\"", 1);
}

static void sb_base64(sbuf *b, const unsigned char *d, size_t n)
{
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    sb_reserve(b, (n + 2) / 3 * 4 + 2);
    char *o = b->p + b->n;
    size_t i = 0;
    for (; i + 2 < n; i += 3) {
        unsigned v = (unsigned)d[i] << 16 | (unsigned)d[i + 1] << 8 | d[i + 2];
        *o++ = tab[v >> 18]; *o++ = tab[(v >> 12) & 63]; *o++ = tab[(v >> 6) & 63]; *o++ = tab[v & 63];
    }
    if (i < n) {
        unsigned v = (unsigned)d[i] << 16 | (i + 1 < n ? (unsigned)d[i + 1] << 8 : 0);
        *o++ = tab[v >> 18]; *o++ = tab[(v >> 12) & 63];
        *o++ = i + 1 < n ? tab[(v >> 6) & 63] : '=';
        *o++ = '=';
    }
    b->n = (size_t)(o - b->p);
    b->p[b->n] = 0;
}

/* ---- State ---------------------------------------------------------------- */

typedef struct { uint32_t stamp; int32_t writer; int32_t reader; int8_t atomic; } cell_t;

typedef struct {
    int id;
    uintptr_t addr;
    size_t size;
    unsigned char *data;
    unsigned char *init;          /* 1 per byte that was ever written        */
    cell_t *cells;                /* race shadow, one per element            */
    size_t ncells;
    size_t gran;
    int no_race;
    int elem_size, type;
    char name[64];
    const char *file;
    int line, freed, free_line;
    int touched_launch;
    int wr_launch;                /* coverage of the current launch          */
    long long wr_count, wr_max;
} alloc_t;

typedef struct { void *p; size_t size; } pinned_t;

enum {
    KD_NULL, KD_HOST_MEMORY, KD_NOT_ALLOC, KD_USE_AFTER_FREE, KD_OOB_WRITE, KD_OOB_READ,
    KD_UNINIT_READ, KD_RACE_WRITE, KD_RACE_READ, KD_KINDS
};

typedef struct {
    int kind, alloc, line;
    const char *name, *file;
    long long count, threads, last_thread;
    long long first_thread, first_index, min_index, max_index, other_thread;
    uintptr_t addr;
} kdiag_t;

enum { F_OOB = 8, F_RACE = 16, F_UNINIT = 32, F_INVALID = 64, F_ATOMIC = 128 };

typedef struct { int32_t thread; int32_t alloc; long long index; int32_t flags; } rec_t;

typedef struct { const char *label; char text[512]; } row_t;

static struct {
    int initialized, finalized, color, tracing, shuffle, strict;
    uint64_t seed;
    uintptr_t region, region_size, bump;
    size_t page;
    alloc_t **allocs;
    int nallocs, cap_allocs;
    size_t used, peak, limit;
    pinned_t *pinned;
    int npinned, cap_pinned;

    cudaError last_error, sticky;
    int last_error_line, last_error_checked, sticky_line, sticky_announced, launch_fail_line;
    const char *last_error_file, *launch_fail_file;

    double host_ms, busy_ms, last_real;
    double kernel_ms, h2d_ms, d2h_ms, wait_ms;
    long long threads_run;
    int nlaunches;

    sbuf events;
    int nevents, events_dropped;
    sbuf diags;
    int ndiags, nerrors, nwarnings, printed;
    size_t snap_total;
    long long rec_total;

    const char *srcs[32];
    int nsrcs;

    uintptr_t stack_lo, stack_hi;
    char exe[PATH_MAX];
    char trace_path[PATH_MAX];
    const char *in_api;
    const char *api_file;
    int api_line;
    int crashed;
} G;

static struct {
    int active, failed, id;
    const char *kernel, *file, *kfile;
    int line, kfirst, klast;
    dim3 grid, block;
    long long total, per_block, cur_thread, ops, thread_recs, total_ops, max_ops;
    int half_bits;
    uint64_t key;
    kdiag_t kd[MAX_KDIAG];
    int nkd;
    rec_t *rec;
    size_t nrec, caprec;
    long long rec_dropped;
    long long *order;
    size_t norder, caporder;
    double t_host;
    char reason[512];
} C;

/* ---- Small helpers ---------------------------------------------------------- */

#define COL(code) (G.color ? code : "")
#define RED    COL("\033[1;31m")
#define YELLOW COL("\033[1;33m")
#define CYAN   COL("\033[1;36m")
#define BLUE   COL("\033[1;34m")
#define BOLD   COL("\033[1m")
#define DIM    COL("\033[2m")
#define RESET  COL("\033[0m")

static double now_real_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static double max_d(double a, double b) { return a > b ? a : b; }

static void fmt_bytes(size_t n, char *buf, size_t len)
{
    if (n < 1024) snprintf(buf, len, "%zu bytes", n);
    else if (n < 1024 * 1024) snprintf(buf, len, "%.1f KB", (double)n / 1024);
    else if (n < 1024u * 1024 * 1024) snprintf(buf, len, "%.1f MB", (double)n / (1024 * 1024));
    else snprintf(buf, len, "%.2f GB", (double)n / (1024.0 * 1024 * 1024));
}

static void fmt_int(long long v, char *buf, size_t len)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%lld", v < 0 ? -v : v);
    size_t o = 0;
    if (v < 0 && o + 1 < len) buf[o++] = '-';
    for (int i = 0; i < n && o + 2 < len; i++) {
        if (i && (n - i) % 3 == 0) buf[o++] = ',';
        buf[o++] = tmp[i];
    }
    buf[o] = 0;
}

static const char *type_name(int t)
{
    static const char *names[] = {
        "byte", "float", "double", "char", "unsigned char", "short", "unsigned short", "int",
        "unsigned int", "long", "unsigned long", "long long", "unsigned long long", "bool",
    };
    return t >= 0 && t < 14 ? names[t] : "byte";
}

static const char *rel_path(const char *path)
{
    static char cwd[PATH_MAX];
    static int have;
    if (!path) return "?";
    if (!have) { have = getcwd(cwd, sizeof cwd) ? 1 : -1; }
    size_t n = have > 0 ? strlen(cwd) : 0;
    if (n && strncmp(path, cwd, n) == 0 && path[n] == '/') return path + n + 1;
    return path;
}

/* "(void **)&d_a" -> "d_a" */
static void clean_name(const char *expr, char *out, size_t len)
{
    const char *s = expr ? expr : "?";
    for (;;) {
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '(') {
            const char *close = strchr(s, ')');
            if (close && memchr(s, '*', (size_t)(close - s))) { s = close + 1; continue; }
        }
        if (*s == '&') { s++; continue; }
        break;
    }
    size_t o = 0;
    for (; *s && o + 1 < len; s++)
        if (*s != ' ' || (o && out[o - 1] != ' ')) out[o++] = *s;
    while (o && out[o - 1] == ' ') o--;
    out[o] = 0;
    if (o > 2 && out[0] == '(' && out[o - 1] == ')' && !strchr(out + 1, '(')) {
        memmove(out, out + 1, o - 2);
        out[o - 2] = 0;
    }
}

static void note_source(const char *file)
{
    if (!file) return;
    for (int i = 0; i < G.nsrcs; i++)
        if (strcmp(G.srcs[i], file) == 0) return;
    if (G.nsrcs < 32) G.srcs[G.nsrcs++] = file;
}

/* Source file cache, for printing the offending line. */
typedef struct { const char *path; char *text; size_t *starts; int nlines; } srcfile_t;
static srcfile_t srccache[16];
static int nsrccache;

static srcfile_t *source_get(const char *path)
{
    for (int i = 0; i < nsrccache; i++)
        if (strcmp(srccache[i].path, path) == 0) return srccache[i].text ? &srccache[i] : NULL;
    if (nsrccache == 16) return NULL;
    srcfile_t *s = &srccache[nsrccache++];
    s->path = path;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    sbuf b = {0};
    char chunk[8192];
    size_t k;
    while ((k = fread(chunk, 1, sizeof chunk, f)) > 0) sb_putn(&b, chunk, k);
    fclose(f);
    if (!b.p) sb_puts(&b, "");
    s->text = b.p;
    int cap = 64;
    s->starts = malloc(sizeof(size_t) * (size_t)cap);
    s->starts[s->nlines++] = 0;
    for (size_t i = 0; i < b.n; i++) {
        if (b.p[i] != '\n') continue;
        if (s->nlines == cap) { cap *= 2; s->starts = realloc(s->starts, sizeof(size_t) * (size_t)cap); }
        s->starts[s->nlines++] = i + 1;
    }
    return s;
}

static int source_line(const char *path, int line, const char **text)
{
    srcfile_t *s = path ? source_get(path) : NULL;
    if (!s || line < 1 || line > s->nlines) return -1;
    const char *p = s->text + s->starts[line - 1];
    int n = 0;
    while (p[n] && p[n] != '\n' && p[n] != '\r') n++;
    *text = p;
    return n;
}

/* ---- Diagnostics ------------------------------------------------------------- */

enum { SEV_NOTE, SEV_WARNING, SEV_ERROR };

static void report(int sev, const char *code, const char *file, int line, int alloc_id,
                   const char *title, const row_t *rows, int nrows, const char *hint)
{
    if (sev == SEV_ERROR) G.nerrors++;
    if (sev == SEV_WARNING) G.nwarnings++;
    note_source(file);

    if (G.printed < MAX_DIAG_PRINTED) {
        const char *sevname = sev == SEV_ERROR ? "error" : sev == SEV_WARNING ? "warning" : "note";
        const char *sevcol = sev == SEV_ERROR ? RED : sev == SEV_WARNING ? YELLOW : CYAN;
        fprintf(stderr, "\n%scuemu %s%s%s[%s]%s: %s%s%s\n", sevcol, sevname, RESET, DIM, code, RESET,
                BOLD, title, RESET);
        if (file && line > 0) {
            fprintf(stderr, "   %s-->%s %s:%d\n", BLUE, RESET, rel_path(file), line);
            const char *text;
            int n = source_line(file, line, &text);
            if (n >= 0) {
                while (n > 0 && (*text == ' ' || *text == '\t')) { text++; n--; }
                fprintf(stderr, "%s%5d |%s %.*s\n", BLUE, line, RESET, n, text);
            }
        }
        for (int i = 0; i < nrows; i++)
            fprintf(stderr, "      %s=%s %-8s %s\n", BLUE, RESET, rows[i].label, rows[i].text);
        if (hint) fprintf(stderr, "      %s= hint%s     %s\n", CYAN, RESET, hint);
    } else if (G.printed == MAX_DIAG_PRINTED) {
        fprintf(stderr, "\n%scuemu: too many problems; the rest are only in the visualization.%s\n", DIM, RESET);
    }
    G.printed++;

    if (G.ndiags >= MAX_DIAG_JSON) return;
    if (G.ndiags) sb_puts(&G.diags, ",\n");
    G.ndiags++;
    sb_printf(&G.diags, "{\"severity\":\"%s\",\"code\":",
              sev == SEV_ERROR ? "error" : sev == SEV_WARNING ? "warning" : "note");
    sb_json_str(&G.diags, code);
    sb_puts(&G.diags, ",\"title\":");
    sb_json_str(&G.diags, title);
    sb_puts(&G.diags, ",\"file\":");
    if (file) sb_json_str(&G.diags, file); else sb_puts(&G.diags, "null");
    sb_printf(&G.diags, ",\"line\":%d,\"alloc\":%d,\"event\":%d,\"launch\":%d,\"rows\":[",
              line, alloc_id, G.nevents, C.active || C.failed ? C.id : 0);
    for (int i = 0; i < nrows; i++) {
        if (i) sb_puts(&G.diags, ",");
        sb_puts(&G.diags, "[");
        sb_json_str(&G.diags, rows[i].label);
        sb_puts(&G.diags, ",");
        sb_json_str(&G.diags, rows[i].text);
        sb_puts(&G.diags, "]");
    }
    sb_puts(&G.diags, "],\"hint\":");
    if (hint) sb_json_str(&G.diags, hint); else sb_puts(&G.diags, "null");
    sb_puts(&G.diags, "}");
}

#define ROW(r, lbl, ...) do { (r).label = (lbl); snprintf((r).text, sizeof (r).text, __VA_ARGS__); } while (0)

static void set_error(cudaError err, const char *file, int line)
{
    if (err == cudaSuccess) return;
    G.last_error = err;
    G.last_error_file = file;
    G.last_error_line = line;
    G.last_error_checked = 0;
}

/* ---- Events ------------------------------------------------------------------ */

static int ev_open(const char *type, const char *file, int line)
{
    if (!G.tracing) return 0;
    if (G.nevents >= MAX_EVENTS) { G.events_dropped++; return 0; }
    if (G.nevents) sb_puts(&G.events, ",\n");
    G.nevents++;
    sb_printf(&G.events, "{\"type\":\"%s\",\"line\":%d,\"file\":", type, line);
    if (file) { sb_json_str(&G.events, file); note_source(file); }
    else sb_puts(&G.events, "null");
    return 1;
}

static void ev_close(void) { sb_puts(&G.events, "}"); }

static void ev_snap(alloc_t *a)
{
    if (!a || a->freed || !a->data) { sb_puts(&G.events, "null"); return; }
    size_t n = a->size < SNAP_MAX_BYTES ? a->size : SNAP_MAX_BYTES;
    if (G.snap_total + n > SNAP_BUDGET) { sb_puts(&G.events, "null"); return; }
    G.snap_total += n;
    sb_printf(&G.events, "{\"alloc\":%d,\"data\":\"", a->id);
    sb_base64(&G.events, a->data, n);
    sb_puts(&G.events, "\",\"init\":\"");
    unsigned char *bits = calloc((n + 7) / 8 + 1, 1);
    for (size_t i = 0; i < n; i++)
        if (a->init[i]) bits[i / 8] |= (unsigned char)(1u << (i % 8));
    sb_base64(&G.events, bits, (n + 7) / 8);
    free(bits);
    sb_puts(&G.events, "\"}");
}

/* ---- Device memory ----------------------------------------------------------- */

static int in_region(uintptr_t p) { return G.region && p >= G.region && p < G.region + G.region_size; }

/* The allocation whose range [addr, addr+size] contains p (one past the end included). */
static alloc_t *find_alloc(uintptr_t p)
{
    int lo = 0, hi = G.nallocs - 1, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (G.allocs[mid]->addr <= p) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (best < 0) return NULL;
    alloc_t *a = G.allocs[best];
    return p <= a->addr + a->size ? a : NULL;
}

static alloc_t *nearest_alloc(uintptr_t p)
{
    alloc_t *best = NULL;
    for (int i = 0; i < G.nallocs && G.allocs[i]->addr <= p; i++) best = G.allocs[i];
    return best;
}

static pinned_t *find_pinned(uintptr_t p)
{
    for (int i = 0; i < G.npinned; i++)
        if (p >= (uintptr_t)G.pinned[i].p && p < (uintptr_t)G.pinned[i].p + G.pinned[i].size) return &G.pinned[i];
    return NULL;
}

static void alloc_desc(const alloc_t *a, char *buf, size_t len)
{
    char sz[32];
    fmt_bytes(a->size, sz, sizeof sz);
    if (a->type && a->elem_size && a->size % (size_t)a->elem_size == 0) {
        char cnt[32];
        fmt_int((long long)(a->size / (size_t)a->elem_size), cnt, sizeof cnt);
        snprintf(buf, len, "`%s`: %s × %s (%s), cudaMalloc at line %d", a->name, cnt, type_name(a->type), sz,
                 a->line);
    } else {
        snprintf(buf, len, "`%s`: %s, cudaMalloc at line %d", a->name, sz, a->line);
    }
}

/* ---- Initialisation ------------------------------------------------------------ */

static void finalize(void);
static void at_exit(void);
static void crash_handler(int sig, siginfo_t *si, void *uctx);

static void ensure_init(void)
{
    if (G.initialized) return;
    G.initialized = 1;
    G.color = isatty(2) && !getenv("NO_COLOR");
    G.tracing = !(getenv("CUEMU_TRACE") && strcmp(getenv("CUEMU_TRACE"), "0") == 0);
    G.strict = getenv("CUEMU_STRICT") && strcmp(getenv("CUEMU_STRICT"), "0") != 0;
    G.shuffle = !(getenv("CUEMU_ORDER") && strcmp(getenv("CUEMU_ORDER"), "sequential") == 0);
    G.seed = getenv("CUEMU_SEED") ? strtoull(getenv("CUEMU_SEED"), NULL, 10) : 0x5eed2026u;
    G.page = (size_t)getpagesize();

    size_t mb = getenv("CUEMU_DEVICE_MB") ? strtoull(getenv("CUEMU_DEVICE_MB"), NULL, 10) : 4096;
    G.limit = (mb ? mb : 4096) << 20;

    for (uintptr_t sz = (uintptr_t)64 << 30; sz >= ((uintptr_t)1 << 30); sz >>= 1) {
        void *p = mmap(NULL, sz, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
        if (p != MAP_FAILED) { G.region = (uintptr_t)p; G.region_size = sz; break; }
    }
    G.bump = G.region + G.page * 16;

#ifdef __APPLE__
    G.stack_hi = (uintptr_t)pthread_get_stackaddr_np(pthread_self());
    G.stack_lo = G.stack_hi - pthread_get_stacksize_np(pthread_self());
    uint32_t len = sizeof G.exe;
    char raw[PATH_MAX];
    if (_NSGetExecutablePath(raw, &len) == 0 && !realpath(raw, G.exe)) snprintf(G.exe, sizeof G.exe, "%s", raw);
#else
    {
        char probe;
        G.stack_hi = (uintptr_t)&probe + 64 * 1024;
        G.stack_lo = G.stack_hi - 8u * 1024 * 1024;
        ssize_t n = readlink("/proc/self/exe", G.exe, sizeof G.exe - 1);
        if (n > 0) G.exe[n] = 0;
    }
#endif

    static char altstack[1 << 16];
    stack_t ss = { .ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0 };
    sigaltstack(&ss, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    atexit(at_exit);

    if (ev_open("context", NULL, 0)) {
        sb_printf(&G.events, ",\"t0\":0,\"t1\":%g", MODEL_CONTEXT_INIT_MS);
        ev_close();
    }
    G.host_ms = MODEL_CONTEXT_INIT_MS;
    G.busy_ms = G.host_ms;
    G.last_real = now_real_ms();
}

static void api_enter(const char *file)
{
    ensure_init();
    note_source(file);
    double now = now_real_ms();
    G.host_ms += now - G.last_real;
    G.last_real = now;
}

static void api_leave(void) { G.last_real = now_real_ms(); }

static cudaError sticky_fail(const char *api, const char *file, int line)
{
    if (!G.sticky_announced) {
        G.sticky_announced = 1;
        row_t r[1];
        ROW(r[0], "cause", "an illegal memory access in the kernel launched at line %d", G.sticky_line);
        char title[160];
        snprintf(title, sizeof title, "%s failed: the device is in an error state", api);
        report(SEV_ERROR, "device-error-state", file, line, -1, title, r, 1,
               "On a real GPU an illegal memory access breaks the CUDA context, and every later "
               "CUDA call returns cudaErrorIllegalAddress. Fix the first error above.");
    }
    set_error(G.sticky, file, line);
    return G.sticky;
}

/* ---- dim3 ----------------------------------------------------------------------- */

dim3 cuemu_dim3_from_int(long long n)
{
    dim3 d = { (unsigned)n, 1, 1 };
    if (n < 0 || n > UINT_MAX) d.x = UINT_MAX;
    return d;
}

dim3 cuemu_dim3_n(const long long *v, size_t n)
{
    dim3 d = { 1, 1, 1 };
    unsigned *f[3] = { &d.x, &d.y, &d.z };
    for (size_t i = 0; i < n && i < 3; i++) *f[i] = (v[i] < 0 || v[i] > UINT_MAX) ? UINT_MAX : (unsigned)v[i];
    return d;
}

/* ---- Memory API ------------------------------------------------------------------ */

cudaError cuemu_malloc(void **devPtr, size_t size, const char *expr, const char *file, int line)
{
    api_enter(file);
    char name[64];
    clean_name(expr, name, sizeof name);
    cudaError err = cudaSuccess;
    alloc_t *a = NULL;
    char title[256];

    if (G.sticky) {
        err = sticky_fail("cudaMalloc", file, line);
    } else if (!devPtr) {
        report(SEV_ERROR, "bad-argument", file, line, -1, "cudaMalloc was given a NULL pointer-to-pointer",
               NULL, 0, "Pass the address of your pointer: cudaMalloc(&d_a, bytes).");
        err = cudaErrorInvalidValue;
    } else if (in_region((uintptr_t)devPtr)) {
        snprintf(title, sizeof title, "cudaMalloc was given the device pointer `%s` instead of its address", name);
        report(SEV_ERROR, "bad-argument", file, line, -1, title, NULL, 0,
               "cudaMalloc stores the new device address into your pointer, so it needs &d_a, not d_a.");
        err = cudaErrorInvalidValue;
    } else if (size > G.limit - G.used || !G.region) {
        char want[32], left[32], total[32];
        fmt_bytes(size, want, sizeof want);
        fmt_bytes(G.limit - G.used, left, sizeof left);
        fmt_bytes(G.limit, total, sizeof total);
        row_t r[1];
        ROW(r[0], "device", "%s free of %s (set CUEMU_DEVICE_MB to change)", left, total);
        snprintf(title, sizeof title, "out of device memory: cudaMalloc(%s) for `%s`", want, name);
        report(SEV_ERROR, "out-of-memory", file, line, -1, title, r, 1,
               "Check the size: cudaMalloc takes bytes, e.g. n * sizeof(float). Also free memory you no longer need.");
        err = cudaErrorMemoryAllocation;
        G.in_api = "cudaMalloc"; G.api_file = file; G.api_line = line;
        *devPtr = NULL;
        G.in_api = NULL;
    } else {
        size_t pages = (size + G.page - 1) / G.page;
        if (G.bump + (pages + 1) * G.page > G.region + G.region_size) G.bump = G.region + G.page * 16;
        a = calloc(1, sizeof *a);
        a->id = G.nallocs + 1;
        a->addr = G.bump;
        a->size = size;
        a->data = malloc(size ? size : 1);
        a->init = calloc(size ? size : 1, 1);
        if (!a->data || !a->init) {
            free(a->data); free(a->init); free(a);
            a = NULL;
            err = cudaErrorMemoryAllocation;
        } else {
            /* Device memory is not zeroed. Fill it with junk so reading it looks wrong. */
            uint64_t x = G.seed ^ (uint64_t)a->id * 0x9E3779B97F4A7C15u;
            for (size_t i = 0; i < size; i++) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                a->data[i] = (unsigned char)x;
            }
            snprintf(a->name, sizeof a->name, "%s", name);
            a->file = file;
            a->line = line;
            G.bump += (pages + 1) * G.page;   /* leave a guard gap after each allocation */
            if (G.nallocs == G.cap_allocs) {
                G.cap_allocs = G.cap_allocs ? G.cap_allocs * 2 : 32;
                G.allocs = realloc(G.allocs, sizeof *G.allocs * (size_t)G.cap_allocs);
            }
            G.allocs[G.nallocs++] = a;
            G.used += size;
            if (G.used > G.peak) G.peak = G.used;
            G.in_api = "cudaMalloc"; G.api_file = file; G.api_line = line;
            *devPtr = (void *)a->addr;
            G.in_api = NULL;
        }
    }

    G.host_ms += MODEL_MALLOC_MS;
    if (ev_open("malloc", file, line)) {
        sb_puts(&G.events, ",\"name\":");
        sb_json_str(&G.events, name);
        sb_printf(&G.events, ",\"bytes\":%zu,\"ok\":%s,\"alloc\":%d,\"t0\":%g,\"t1\":%g", size,
                  err ? "false" : "true", a ? a->id : -1, G.host_ms - MODEL_MALLOC_MS, G.host_ms);
        ev_close();
    }
    set_error(err, file, line);
    api_leave();
    return err;
}

cudaError cuemu_free(void *devPtr, const char *expr, const char *file, int line)
{
    api_enter(file);
    char name[64], title[256], desc[400];
    clean_name(expr, name, sizeof name);
    cudaError err = cudaSuccess;
    uintptr_t p = (uintptr_t)devPtr;
    alloc_t *a = in_region(p) ? find_alloc(p) : NULL;

    if (!devPtr) {
        api_leave();
        return cudaSuccess;
    } else if (G.sticky) {
        err = sticky_fail("cudaFree", file, line);
    } else if (!in_region(p)) {
        snprintf(title, sizeof title, "cudaFree(%s): `%s` is not device memory", name, name);
        report(SEV_ERROR, "free-host-pointer", file, line, -1, title, NULL, 0,
               "cudaFree releases memory from cudaMalloc. Memory from malloc is released with free().");
        err = cudaErrorInvalidValue;
    } else if (!a || a->addr != p) {
        row_t r[1];
        int nr = 0;
        if (a) { alloc_desc(a, desc, sizeof desc); ROW(r[0], "memory", "points %zu bytes into %s", (size_t)(p - a->addr), desc); nr = 1; }
        snprintf(title, sizeof title, "cudaFree(%s): not the start of a device allocation", name);
        report(SEV_ERROR, "free-bad-pointer", file, line, a ? a->id : -1, title, r, nr,
               "Pass exactly the pointer cudaMalloc gave you.");
        err = cudaErrorInvalidValue;
    } else if (a->freed) {
        row_t r[1];
        ROW(r[0], "first", "already freed at line %d", a->free_line);
        snprintf(title, sizeof title, "double free: `%s` was already freed", a->name);
        report(SEV_ERROR, "double-free", file, line, a->id, title, r, 1,
               "Free each allocation exactly once. Setting the pointer to NULL after cudaFree prevents this.");
        err = cudaErrorInvalidValue;
    } else {
        a->freed = 1;
        a->free_line = line;
        G.used -= a->size;
        free(a->data); a->data = NULL;
        free(a->init); a->init = NULL;
        free(a->cells); a->cells = NULL;
    }

    if (ev_open("free", file, line)) {
        sb_puts(&G.events, ",\"name\":");
        sb_json_str(&G.events, name);
        sb_printf(&G.events, ",\"ok\":%s,\"alloc\":%d,\"t0\":%g,\"t1\":%g", err ? "false" : "true",
                  a ? a->id : -1, G.host_ms, G.host_ms + MODEL_API_MS);
        ev_close();
    }
    G.host_ms += MODEL_API_MS;
    set_error(err, file, line);
    api_leave();
    return err;
}

cudaError cuemu_malloc_host(void **ptr, size_t size, const char *expr, const char *file, int line)
{
    api_enter(file);
    (void)expr;
    cudaError err = cudaSuccess;
    void *p = ptr ? malloc(size ? size : 1) : NULL;
    if (!ptr || !p) err = ptr ? cudaErrorMemoryAllocation : cudaErrorInvalidValue;
    else {
        if (G.npinned == G.cap_pinned) {
            G.cap_pinned = G.cap_pinned ? G.cap_pinned * 2 : 16;
            G.pinned = realloc(G.pinned, sizeof *G.pinned * (size_t)G.cap_pinned);
        }
        G.pinned[G.npinned++] = (pinned_t){ p, size };
        *ptr = p;
    }
    G.host_ms += MODEL_MALLOC_MS * 4;   /* pinning pages is slow */
    set_error(err, file, line);
    api_leave();
    return err;
}

cudaError cuemu_free_host(void *ptr, const char *file, int line)
{
    api_enter(file);
    for (int i = 0; i < G.npinned; i++) {
        if (G.pinned[i].p != ptr) continue;
        free(ptr);
        G.pinned[i] = G.pinned[--G.npinned];
        api_leave();
        return cudaSuccess;
    }
    if (ptr) {
        report(SEV_ERROR, "free-host-pointer", file, line, -1,
               "cudaFreeHost was given memory that did not come from cudaMallocHost", NULL, 0, NULL);
        set_error(cudaErrorInvalidValue, file, line);
        api_leave();
        return cudaErrorInvalidValue;
    }
    api_leave();
    return cudaSuccess;
}

enum { P_NULL, P_HOST, P_DEVICE, P_FREED, P_BAD };

typedef struct { int cls; alloc_t *a; size_t off; } ptrinfo;

static ptrinfo classify(const void *ptr)
{
    ptrinfo r = { P_HOST, NULL, 0 };
    uintptr_t p = (uintptr_t)ptr;
    if (!p) { r.cls = P_NULL; return r; }
    if (!in_region(p)) return r;
    r.a = find_alloc(p);
    if (!r.a) { r.cls = P_BAD; r.a = nearest_alloc(p); return r; }
    r.cls = r.a->freed ? P_FREED : P_DEVICE;
    r.off = (size_t)(p - r.a->addr);
    return r;
}

static const char *kind_str(cudaMemcpyKind k)
{
    switch (k) {
    case cudaMemcpyHostToHost: return "HostToHost";
    case cudaMemcpyHostToDevice: return "HostToDevice";
    case cudaMemcpyDeviceToHost: return "DeviceToHost";
    case cudaMemcpyDeviceToDevice: return "DeviceToDevice";
    case cudaMemcpyDefault: return "Default";
    }
    return "invalid";
}

/* Explain why a pointer is on the wrong side. Returns 1 if a problem was reported. */
static int check_side(const ptrinfo *pi, int want_device, const char *role, const char *name,
                      cudaMemcpyKind kind, const char *file, int line)
{
    char title[320], desc[400];
    row_t r[2];
    int nr = 0;
    if (pi->cls == P_NULL) {
        snprintf(title, sizeof title, "cudaMemcpy: the %s `%s` is NULL", role, name);
        report(SEV_ERROR, "memcpy-null", file, line, -1, title, NULL, 0,
               want_device ? "Allocate it with cudaMalloc first, and check the result."
                           : "Allocate the host buffer (e.g. malloc) before copying into or out of it.");
        return 1;
    }
    if (want_device && pi->cls == P_HOST) {
        snprintf(title, sizeof title, "cudaMemcpy%s: the %s `%s` is host memory, not device memory",
                 kind_str(kind), role, name);
        report(SEV_ERROR, "memcpy-wrong-side", file, line, -1, title, NULL, 0,
               "The kind names the direction: HostToDevice means (device dst, host src); "
               "DeviceToHost means (host dst, device src).");
        return 1;
    }
    if (!want_device && (pi->cls == P_DEVICE || pi->cls == P_FREED || pi->cls == P_BAD)) {
        if (pi->a) { alloc_desc(pi->a, desc, sizeof desc); ROW(r[nr], "memory", "%s", desc); nr++; }
        snprintf(title, sizeof title, "cudaMemcpy%s: the %s `%s` is device memory, not host memory",
                 kind_str(kind), role, name);
        report(SEV_ERROR, "memcpy-wrong-side", file, line, pi->a ? pi->a->id : -1, title, r, nr,
               "The kind names the direction: HostToDevice means (device dst, host src); "
               "DeviceToHost means (host dst, device src).");
        return 1;
    }
    if (want_device && pi->cls == P_FREED) {
        ROW(r[0], "freed", "at line %d", pi->a->free_line);
        snprintf(title, sizeof title, "cudaMemcpy uses `%s` after it was freed", pi->a->name);
        report(SEV_ERROR, "use-after-free", file, line, pi->a->id, title, r, 1,
               "Free device memory only after the last copy that uses it.");
        return 1;
    }
    if (want_device && pi->cls == P_BAD) {
        snprintf(title, sizeof title, "cudaMemcpy: `%s` points into device address space but not into any allocation", name);
        report(SEV_ERROR, "bad-device-pointer", file, line, -1, title, NULL, 0,
               "Use the pointer cudaMalloc returned (plus an offset inside the allocation).");
        return 1;
    }
    return 0;
}

static int check_range(const ptrinfo *pi, size_t count, const char *role, const char *file, int line)
{
    if (pi->cls != P_DEVICE || pi->off + count <= pi->a->size) return 0;
    char title[320], desc[400], cnt[32], avail[32];
    alloc_desc(pi->a, desc, sizeof desc);
    fmt_bytes(count, cnt, sizeof cnt);
    fmt_bytes(pi->a->size - pi->off, avail, sizeof avail);
    row_t r[2];
    ROW(r[0], "memory", "%s", desc);
    ROW(r[1], "sizes", "copying %s, but only %s fit starting at the %s pointer", cnt, avail, role);
    snprintf(title, sizeof title, "cudaMemcpy would run past the end of `%s`", pi->a->name);
    report(SEV_ERROR, "memcpy-overflow", file, line, pi->a->id, title, r, 2,
           "cudaMemcpy counts bytes. Use the same byte count you gave cudaMalloc, e.g. n * sizeof(float).");
    return 1;
}

cudaError cuemu_memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                       const char *dst_expr, const char *src_expr, const char *file, int line)
{
    api_enter(file);
    char dname[64], sname[64], title[320];
    clean_name(dst_expr, dname, sizeof dname);
    clean_name(src_expr, sname, sizeof sname);
    ptrinfo D = classify(dst), S = classify(src);
    cudaMemcpyKind eff = kind;
    if (kind == cudaMemcpyDefault) {
        int dd = D.cls == P_DEVICE || D.cls == P_FREED || D.cls == P_BAD;
        int sd = S.cls == P_DEVICE || S.cls == P_FREED || S.cls == P_BAD;
        eff = dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                 : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    }
    cudaError err = cudaSuccess;

    if (G.sticky) {
        err = sticky_fail("cudaMemcpy", file, line);
    } else if ((unsigned)kind > cudaMemcpyDefault) {
        report(SEV_ERROR, "memcpy-bad-kind", file, line, -1, "cudaMemcpy: invalid cudaMemcpyKind", NULL, 0,
               "Use cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost or cudaMemcpyDeviceToDevice.");
        err = cudaErrorInvalidValue;
    } else if ((eff == cudaMemcpyHostToDevice && D.cls == P_HOST && S.cls == P_DEVICE) ||
               (eff == cudaMemcpyDeviceToHost && D.cls == P_DEVICE && S.cls == P_HOST)) {
        row_t r[2];
        ROW(r[0], "dst", "`%s` is %s memory", dname, D.cls == P_DEVICE ? "device" : "host");
        ROW(r[1], "src", "`%s` is %s memory", sname, S.cls == P_DEVICE ? "device" : "host");
        snprintf(title, sizeof title, "cudaMemcpy%s: destination and source are the wrong way round", kind_str(kind));
        report(SEV_ERROR, "memcpy-swapped", file, line, -1, title, r, 2,
               eff == cudaMemcpyHostToDevice
                   ? "The signature is cudaMemcpy(dst, src, bytes, kind). To copy results back, write "
                     "cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost)."
                   : "The signature is cudaMemcpy(dst, src, bytes, kind). To upload data, write "
                     "cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice).");
        err = cudaErrorInvalidValue;
    } else {
        int dev_dst = eff == cudaMemcpyHostToDevice || eff == cudaMemcpyDeviceToDevice;
        int dev_src = eff == cudaMemcpyDeviceToHost || eff == cudaMemcpyDeviceToDevice;
        if (check_side(&D, dev_dst, "destination", dname, kind, file, line) ||
            check_side(&S, dev_src, "source", sname, kind, file, line) ||
            check_range(&D, count, "destination", file, line) ||
            check_range(&S, count, "source", file, line))
            err = cudaErrorInvalidValue;
    }

    int pinned = 0;
    if (!err) {
        const ptrinfo *dev = D.cls == P_DEVICE ? &D : S.cls == P_DEVICE ? &S : NULL;
        if (dev && dev->off == 0 && count < dev->a->size && count > 0 &&
            (count * 4 == dev->a->size || count * 8 == dev->a->size)) {
            char want[32], have[32];
            fmt_bytes(count, want, sizeof want);
            fmt_bytes(dev->a->size, have, sizeof have);
            row_t r[1];
            ROW(r[0], "sizes", "copying %s, but `%s` is %s", want, dev->a->name, have);
            snprintf(title, sizeof title, "cudaMemcpy copies only 1/%zu of `%s`", dev->a->size / count, dev->a->name);
            report(SEV_WARNING, "memcpy-partial", file, line, dev->a->id, title, r, 1,
                   "The count looks like a number of elements. cudaMemcpy counts bytes: n * sizeof(float).");
        }
        G.in_api = "cudaMemcpy"; G.api_file = file; G.api_line = line;
        if (eff == cudaMemcpyHostToDevice) {
            memcpy(D.a->data + D.off, src, count);
            memset(D.a->init + D.off, 1, count);
            pinned = find_pinned((uintptr_t)src) != NULL;
        } else if (eff == cudaMemcpyDeviceToHost) {
            size_t unset = 0;
            for (size_t i = 0; i < count; i++) unset += !S.a->init[S.off + i];
            if (unset) {
                char u[32], total[32];
                row_t r[1];
                if (S.a->elem_size && count % (size_t)S.a->elem_size == 0) {
                    size_t bad = 0, es = (size_t)S.a->elem_size;
                    for (size_t e = 0; e < count / es; e++) bad += !S.a->init[S.off + e * es];
                    fmt_int((long long)bad, u, sizeof u);
                    fmt_int((long long)(count / es), total, sizeof total);
                    ROW(r[0], "memory", "%s of the %s %s values copied were never written on the device", u, total,
                        type_name(S.a->type));
                } else {
                    fmt_int((long long)unset, u, sizeof u);
                    fmt_int((long long)count, total, sizeof total);
                    ROW(r[0], "memory", "%s of the %s copied bytes were never written on the device", u, total);
                }
                snprintf(title, sizeof title, "copying uninitialized device memory from `%s` back to `%s`", S.a->name, dname);
                report(SEV_WARNING, "memcpy-uninitialized", file, line, S.a->id, title, r, 1,
                       "No cudaMemcpy or kernel ever wrote those bytes, so the host gets garbage. "
                       "Did every element get a GPU thread? Did the kernel launch fail?");
            }
            memcpy(dst, S.a->data + S.off, count);
            pinned = find_pinned((uintptr_t)dst) != NULL;
        } else if (eff == cudaMemcpyDeviceToDevice) {
            memmove(D.a->data + D.off, S.a->data + S.off, count);
            memmove(D.a->init + D.off, S.a->init + S.off, count);
        } else {
            memmove(dst, src, count);
        }
        G.in_api = NULL;
    }

    double h0 = G.host_ms;
    double t0 = max_d(G.host_ms, G.busy_ms);
    double dur = 0;
    if (eff == cudaMemcpyHostToHost) dur = (double)count / 1e10 * 1e3;
    else if (eff == cudaMemcpyDeviceToDevice) dur = MODEL_PCIE_LATENCY_MS + (double)count / 5e10 * 1e3;
    else dur = MODEL_PCIE_LATENCY_MS +
               (double)count / ((pinned ? MODEL_PCIE_PINNED_GBPS : MODEL_PCIE_PAGEABLE_GBPS) * 1e9) * 1e3;
    if (err) dur = MODEL_API_MS;
    G.wait_ms += t0 - h0;
    G.host_ms = G.busy_ms = t0 + dur;
    if (!err && eff == cudaMemcpyHostToDevice) G.h2d_ms += dur;
    if (!err && eff == cudaMemcpyDeviceToHost) G.d2h_ms += dur;

    if (ev_open("memcpy", file, line)) {
        sb_printf(&G.events, ",\"kind\":\"%s\",\"ok\":%s,\"bytes\":%zu,\"pinned\":%s", kind_str(eff),
                  err ? "false" : "true", count, pinned ? "true" : "false");
        sb_puts(&G.events, ",\"dst\":");
        sb_json_str(&G.events, dname);
        sb_puts(&G.events, ",\"src\":");
        sb_json_str(&G.events, sname);
        sb_printf(&G.events, ",\"dstAlloc\":%d,\"srcAlloc\":%d,\"tHost\":%g,\"t0\":%g,\"t1\":%g,\"snap\":",
                  D.a && D.cls == P_DEVICE ? D.a->id : -1, S.a && S.cls == P_DEVICE ? S.a->id : -1, h0, t0,
                  t0 + dur);
        alloc_t *snap = !err && D.cls == P_DEVICE ? D.a : !err && S.cls == P_DEVICE ? S.a : NULL;
        ev_snap(snap);
        ev_close();
    }
    set_error(err, file, line);
    api_leave();
    return err;
}

cudaError cuemu_memset(void *devPtr, int value, size_t count, const char *expr, const char *file, int line)
{
    api_enter(file);
    char name[64];
    clean_name(expr, name, sizeof name);
    cudaError err = cudaSuccess;
    ptrinfo P = classify(devPtr);
    if (G.sticky) err = sticky_fail("cudaMemset", file, line);
    else if (check_side(&P, 1, "pointer", name, cudaMemcpyHostToDevice, file, line) ||
             check_range(&P, count, "", file, line))
        err = cudaErrorInvalidValue;
    else {
        memset(P.a->data + P.off, value, count);
        memset(P.a->init + P.off, 1, count);
    }
    double t0 = max_d(G.host_ms, G.busy_ms);
    G.busy_ms = t0 + MODEL_LAUNCH_MS;
    G.host_ms += MODEL_API_MS;
    if (ev_open("memset", file, line)) {
        sb_puts(&G.events, ",\"name\":");
        sb_json_str(&G.events, name);
        sb_printf(&G.events, ",\"ok\":%s,\"bytes\":%zu,\"value\":%d,\"alloc\":%d,\"t0\":%g,\"t1\":%g,\"snap\":",
                  err ? "false" : "true", count, value, P.a ? P.a->id : -1, t0, G.busy_ms);
        ev_snap(err ? NULL : P.a);
        ev_close();
    }
    set_error(err, file, line);
    api_leave();
    return err;
}

cudaError cuemu_synchronize(const char *file, int line)
{
    api_enter(file);
    double h0 = G.host_ms;
    G.host_ms = max_d(G.host_ms, G.busy_ms);
    G.wait_ms += G.host_ms - h0;
    if (ev_open("sync", file, line)) {
        sb_printf(&G.events, ",\"t0\":%g,\"t1\":%g", h0, G.host_ms);
        ev_close();
    }
    api_leave();
    if (G.sticky) { set_error(G.sticky, file, line); return G.sticky; }
    return cudaSuccess;
}

cudaError cuemu_get_last_error(int peek, const char *file, int line)
{
    api_enter(file);
    (void)line;
    cudaError e = G.last_error;
    G.last_error_checked = 1;
    G.launch_fail_line = 0;
    if (!peek && !G.sticky) G.last_error = cudaSuccess;
    api_leave();
    return e;
}

cudaError cuemu_mem_get_info(size_t *free_bytes, size_t *total_bytes)
{
    ensure_init();
    if (free_bytes) *free_bytes = G.limit - G.used;
    if (total_bytes) *total_bytes = G.limit;
    return cudaSuccess;
}

cudaError cuemu_device_reset(const char *file, int line)
{
    api_enter(file);
    for (int i = 0; i < G.nallocs; i++) {
        alloc_t *a = G.allocs[i];
        if (a->freed) continue;
        a->freed = 1;
        a->free_line = line;
        free(a->data); a->data = NULL;
        free(a->init); a->init = NULL;
        free(a->cells); a->cells = NULL;
    }
    G.used = 0;
    G.sticky = cudaSuccess;
    G.sticky_announced = 0;
    G.last_error = cudaSuccess;
    G.host_ms = max_d(G.host_ms, G.busy_ms);
    api_leave();
    return cudaSuccess;
}

const char *cudaGetErrorName(cudaError e)
{
    switch (e) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
    }
    return "cudaErrorUnknown";
}

const char *cudaGetErrorString(cudaError e)
{
    switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorInvalidConfiguration: return "invalid configuration argument";
    case cudaErrorInvalidDevice: return "invalid device ordinal";
    case cudaErrorIllegalAddress: return "an illegal memory access was encountered";
    }
    return "unknown error";
}

cudaError cudaGetDeviceCount(int *count) { if (count) *count = 1; return cudaSuccess; }
cudaError cudaSetDevice(int device) { return device == 0 ? cudaSuccess : cudaErrorInvalidDevice; }
cudaError cudaGetDevice(int *device) { if (device) *device = 0; return cudaSuccess; }

/* ---- Kernel launches ------------------------------------------------------------- */

static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15u;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9u;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBu;
    return x ^ (x >> 31);
}

/* A random permutation of [0, total) without storing it: Feistel + cycle walking. */
static long long permute(long long i)
{
    uint64_t mask = ((uint64_t)1 << C.half_bits) - 1;
    uint64_t x = (uint64_t)i;
    do {
        uint64_t l = x >> C.half_bits, r = x & mask;
        for (uint64_t round = 0; round < 4; round++) {
            uint64_t f = mix64(r ^ C.key ^ (round * 0x632BE59BD9B4E019u)) & mask;
            uint64_t nl = r;
            r = l ^ f;
            l = nl;
        }
        x = (l << C.half_bits) | r;
    } while (x >= (uint64_t)C.total);
    return (long long)x;
}

static void thread_pos(long long t, char *buf, size_t len)
{
    long long b = t / C.per_block, th = t % C.per_block;
    unsigned bx = (unsigned)(b % C.grid.x), by = (unsigned)(b / C.grid.x % C.grid.y), bz = (unsigned)(b / ((long long)C.grid.x * C.grid.y));
    unsigned tx = (unsigned)(th % C.block.x), ty = (unsigned)(th / C.block.x % C.block.y), tz = (unsigned)(th / ((long long)C.block.x * C.block.y));
    char id[32];
    fmt_int(t, id, sizeof id);
    if (C.grid.y == 1 && C.grid.z == 1 && C.block.y == 1 && C.block.z == 1)
        snprintf(buf, len, "GPU thread #%s: blockIdx.x = %u, threadIdx.x = %u", id, bx, tx);
    else if (C.grid.z == 1 && C.block.z == 1)
        snprintf(buf, len, "GPU thread #%s: blockIdx = (%u, %u), threadIdx = (%u, %u)", id, bx, by, tx, ty);
    else
        snprintf(buf, len, "GPU thread #%s: blockIdx = (%u, %u, %u), threadIdx = (%u, %u, %u)", id, bx, by, bz, tx, ty, tz);
}

static const char *config_problem(dim3 grid, dim3 block, char *hint, size_t hlen)
{
    static char msg[256];
    hint[0] = 0;
    if (!block.x || !block.y || !block.z) {
        snprintf(msg, sizeof msg, "threadsPerBlock is (%u, %u, %u): every component must be at least 1", block.x, block.y, block.z);
        snprintf(hint, hlen, "%s", block.x && block.y
            ? "In C, `dim3 b = {16, 16};` leaves z = 0. Write {16, 16, 1}, or dim3 b(16, 16)."
            : "If you computed this with integer division, it may have rounded down to 0.");
        return msg;
    }
    if (!grid.x || !grid.y || !grid.z) {
        snprintf(msg, sizeof msg, "numBlocks is (%u, %u, %u): every component must be at least 1", grid.x, grid.y, grid.z);
        snprintf(hint, hlen, "%s", grid.x && grid.y
            ? "In C, `dim3 g = {4, 4};` leaves z = 0. Write {4, 4, 1}, or dim3 g(4, 4)."
            : "n / threadsPerBlock rounds down to 0 when n < threadsPerBlock. Round up instead: "
              "(n + threadsPerBlock - 1) / threadsPerBlock.");
        return msg;
    }
    unsigned long long per = (unsigned long long)block.x * block.y * block.z;
    if (block.x > 1024 || block.y > 1024 || block.z > 1024 || per > MAX_THREADS_PER_BLOCK) {
        snprintf(msg, sizeof msg, "threadsPerBlock (%u, %u, %u) is %llu threads; a block holds at most %d",
                 block.x, block.y, block.z, per, MAX_THREADS_PER_BLOCK);
        if (grid.x <= 1024 && grid.y == 1 && block.y == 1)
            snprintf(hint, hlen, "Launch syntax is kernel<<<numBlocks, threadsPerBlock>>>: the block count comes first. "
                                 "Did you swap them? Try <<<(n + 255) / 256, 256>>>.");
        else
            snprintf(hint, hlen, "Use more blocks with fewer threads each, e.g. 16 x 16 = 256 threads per block.");
        return msg;
    }
    if (grid.y > 65535 || grid.z > 65535 || grid.x > 2147483647u) {
        snprintf(msg, sizeof msg, "numBlocks (%u, %u, %u) is too large (y and z are limited to 65535)", grid.x, grid.y, grid.z);
        snprintf(hint, hlen, "Put the long dimension in x, which allows up to 2^31 - 1 blocks.");
        return msg;
    }
    return NULL;
}

int cuemu_launch_begin(cuemu_launch *L, const char *kernel, dim3 grid, dim3 block, const char *file, int line)
{
    api_enter(file);
    memset(L, 0, sizeof *L);
    L->id = ++G.nlaunches;

    C.active = C.failed = 0;
    C.id = L->id;
    C.kernel = kernel;
    C.file = file;
    C.line = line;
    C.grid = grid;
    C.block = block;
    C.kfile = NULL;
    C.kfirst = C.klast = 0;
    C.nkd = 0;
    C.nrec = C.norder = 0;
    C.rec_dropped = 0;
    C.ops = C.total_ops = C.max_ops = 0;
    C.t_host = G.host_ms;
    C.reason[0] = 0;

    char hint[256];
    const char *problem = config_problem(grid, block, hint, sizeof hint);
    if (G.sticky) {
        sticky_fail("kernel launch", file, line);
        snprintf(C.reason, sizeof C.reason, "not run: device is in an error state");
        C.failed = 1;
        return 0;
    }
    if (problem) {
        C.failed = 1;
        char title[320];
        snprintf(title, sizeof title, "invalid launch configuration for kernel `%s`", kernel);
        row_t r[2];
        ROW(r[0], "problem", "%s", problem);
        ROW(r[1], "result", "the kernel did not run; cudaGetLastError() returns cudaErrorInvalidConfiguration");
        report(SEV_ERROR, "invalid-configuration", file, line, -1, title, r, 2, hint);
        snprintf(C.reason, sizeof C.reason, "%s", problem);
        set_error(cudaErrorInvalidConfiguration, file, line);
        G.launch_fail_file = file;
        G.launch_fail_line = line;
        return 0;
    }

    C.per_block = (long long)block.x * block.y * block.z;
    C.total = (long long)grid.x * grid.y * grid.z * C.per_block;
    int bits = 2;
    while (bits < 62 && ((long long)1 << bits) < C.total) bits++;
    C.half_bits = (bits + 1) / 2;
    C.key = mix64(G.seed + (uint64_t)C.id);
    C.active = 1;
    L->ok = 1;
    L->total = C.total;
    return 1;
}

static void finish_thread(void)
{
    long long ops = C.ops + 1;
    C.total_ops += ops;
    if (ops > C.max_ops) C.max_ops = ops;
}

int cuemu_launch_next(cuemu_launch *L)
{
    if (!L->ok) return 0;
    if (L->next > 0) finish_thread();
    if (L->next >= L->total) return 0;
    long long t = G.shuffle ? permute(L->next) : L->next;
    L->next++;

    long long b = t / C.per_block, th = t % C.per_block;
    cuemu_tls.block_idx = (dim3){ (unsigned)(b % C.grid.x), (unsigned)(b / C.grid.x % C.grid.y),
                                  (unsigned)(b / ((long long)C.grid.x * C.grid.y)) };
    cuemu_tls.thread_idx = (dim3){ (unsigned)(th % C.block.x), (unsigned)(th / C.block.x % C.block.y),
                                   (unsigned)(th / ((long long)C.block.x * C.block.y)) };
    cuemu_tls.block_dim = C.block;
    cuemu_tls.grid_dim = C.grid;
    cuemu_tls.thread_id = t;
    cuemu_tls.in_kernel = 1;
    cuemu_tls.kernel_fp = NULL;
    C.cur_thread = t;
    C.ops = 0;
    C.thread_recs = 0;

    if (G.tracing && C.total <= REC_MAX_THREAD_ID) {
        if (C.norder == C.caporder) {
            C.caporder = C.caporder ? C.caporder * 2 : 4096;
            C.order = realloc(C.order, sizeof *C.order * C.caporder);
        }
        C.order[C.norder++] = t;
    }
    return 1;
}

void cuemu_kernel_enter(void *frame, const char *kernel, int first_line, int last_line)
{
    (void)kernel;
    cuemu_tls.kernel_fp = frame;
    if (!C.kfirst) { C.kfirst = first_line; C.klast = last_line; }
}

static void kdiag(int kind, alloc_t *a, const char *name, const char *file, int line, long long index,
                  long long other, uintptr_t addr)
{
    int aid = a ? a->id : -1;
    long long t = cuemu_tls.thread_id;
    kdiag_t *d = NULL;
    for (int i = 0; i < C.nkd; i++) {
        kdiag_t *k = &C.kd[i];
        if (k->kind == kind && k->alloc == aid && k->line == line && strcmp(k->name, name) == 0) { d = k; break; }
    }
    if (!d) {
        if (C.nkd == MAX_KDIAG) return;
        d = &C.kd[C.nkd++];
        memset(d, 0, sizeof *d);
        d->kind = kind; d->alloc = aid; d->line = line; d->name = name; d->file = file;
        d->first_thread = t; d->first_index = d->min_index = d->max_index = index;
        d->other_thread = other; d->addr = addr; d->last_thread = -1;
    }
    d->count++;
    if (index < d->min_index) d->min_index = index;
    if (index > d->max_index) d->max_index = index;
    if (d->last_thread != t) { d->threads++; d->last_thread = t; }
}

static void rec_add(int alloc, long long index, int flags)
{
    if (!G.tracing || !C.active) return;
    long long t = cuemu_tls.thread_id;
    if (t >= REC_MAX_THREAD_ID || G.rec_total >= REC_MAX_TOTAL || C.thread_recs >= REC_MAX_PER_THREAD) {
        C.rec_dropped++;
        return;
    }
    if (C.nrec == C.caprec) {
        C.caprec = C.caprec ? C.caprec * 2 : 4096;
        C.rec = realloc(C.rec, sizeof *C.rec * C.caprec);
    }
    C.rec[C.nrec++] = (rec_t){ (int32_t)t, alloc, index, flags };
    C.thread_recs++;
    G.rec_total++;
}

static void *scratch(size_t n)
{
    static _Thread_local unsigned char small[256];
    static unsigned char *big;
    static size_t bigsize;
    if (n <= sizeof small) { memset(small, 0, n); return small; }
    if (n > bigsize) { free(big); big = calloc(1, n); bigsize = n; }
    else memset(big, 0, n);
    return big;
}

void *cuemu_access(void *base, size_t esz, long long index, int rw, int type, const char *name,
                   const char *file, int line)
{
    uintptr_t b = (uintptr_t)base;
    int in_kernel = cuemu_tls.in_kernel;
    int atomic = rw & CUEMU_ATOMIC;
    rw &= CUEMU_RW | CUEMU_ADDR;

    if (!in_region(b)) {
        if (!in_kernel || rw == CUEMU_ADDR) return (char *)base + index * (long long)esz;
        if (cuemu_tls.kernel_fp && b >= G.stack_lo && b < (uintptr_t)cuemu_tls.kernel_fp)
            return (char *)base + index * (long long)esz;           /* a kernel-local variable */
        C.ops++;
        if (b < 65536) {
            kdiag(KD_NULL, NULL, name, file, line, index, -1, b);
        } else {
            kdiag(KD_HOST_MEMORY, NULL, name, file, line, index, -1, b);
            if (!G.sticky) { G.sticky = cudaErrorIllegalAddress; G.sticky_line = C.line; }
        }
        rec_add(-1, index, rw | F_INVALID);
        return scratch(esz);
    }

    alloc_t *a = find_alloc(b);
    if (!in_kernel) {
        if (rw == CUEMU_ADDR) return (char *)base + index * (long long)esz;
        static int reported_line;
        if (reported_line != line) {
            reported_line = line;
            char title[256], desc[400];
            row_t r[1];
            int nr = 0;
            if (a) { alloc_desc(a, desc, sizeof desc); ROW(r[0], "memory", "%s", desc); nr = 1; }
            snprintf(title, sizeof title, "host code used device memory through `%s`", name);
            report(SEV_ERROR, "device-pointer-on-host", file, line, a ? a->id : -1, title, r, nr,
                   "This __device__ function was called from the CPU with a device pointer. "
                   "Copy the data back with cudaMemcpy(..., cudaMemcpyDeviceToHost) and use the host copy.");
        }
        return scratch(esz);
    }

    C.ops++;
    if (!a) {
        kdiag(KD_NOT_ALLOC, nearest_alloc(b), name, file, line, index, -1, b);
        rec_add(-1, index, rw | F_INVALID);
        return scratch(esz);
    }
    if (a->freed) {
        kdiag(KD_USE_AFTER_FREE, a, name, file, line, index, -1, b);
        rec_add(a->id, index, rw | F_INVALID);
        return scratch(esz);
    }

    long long off = (long long)(b - a->addr) + index * (long long)esz;
    if (rw == CUEMU_ADDR) return (void *)(b + (uintptr_t)(index * (long long)esz));
    if (!a->type) { a->type = type; a->elem_size = (int)esz; }
    long long ei = off >= 0 ? off / (long long)esz : -((-off + (long long)esz - 1) / (long long)esz);
    a->touched_launch = C.id;

    if (off < 0 || (unsigned long long)off + esz > a->size) {
        kdiag((rw & CUEMU_WRITE) ? KD_OOB_WRITE : KD_OOB_READ, a, name, file, line, ei, -1, b);
        rec_add(a->id, ei, rw | F_OOB);
        return scratch(esz);
    }

    int flags = rw | (atomic ? F_ATOMIC : 0);
    if ((rw & CUEMU_READ) && !a->init[off]) {
        kdiag(KD_UNINIT_READ, a, name, file, line, ei, -1, b);
        flags |= F_UNINIT;
    }

    if (!a->cells && !a->no_race) {
        size_t n = a->size / esz + 1;
        if (n > ((size_t)1 << 26) || !(a->cells = calloc(n, sizeof(cell_t)))) a->no_race = 1;
        else { a->ncells = n; a->gran = esz; }
    }
    size_t ci = a->cells ? (size_t)off / a->gran : 0;
    if (a->cells && ci < a->ncells) {
        cell_t *c = &a->cells[ci];
        int32_t t = (int32_t)cuemu_tls.thread_id;
        if (c->stamp != (uint32_t)C.id) { c->stamp = (uint32_t)C.id; c->writer = -1; c->reader = -1; c->atomic = 0; }
        int raced = 0;
        int safe = atomic && (c->atomic || c->writer < 0);
        if (safe) {
            /* atomics are allowed to share an element */
        } else if ((rw & CUEMU_WRITE) && c->writer >= 0 && c->writer != t) {
            kdiag(KD_RACE_WRITE, a, name, file, line, ei, c->writer, b);
            raced = 1;
        } else if (c->writer >= 0 && c->writer != t) {
            kdiag(KD_RACE_READ, a, name, file, line, ei, c->writer, b);
            raced = 1;
        } else if ((rw & CUEMU_WRITE) && (c->reader == -2 || (c->reader >= 0 && c->reader != t))) {
            kdiag(KD_RACE_READ, a, name, file, line, ei, c->reader, b);
            raced = 1;
        }
        if (raced) flags |= F_RACE;
        if (atomic) c->atomic = 1;
        if (rw & CUEMU_READ) c->reader = c->reader == -1 || c->reader == t ? t : -2;
        if (rw & CUEMU_WRITE) {
            if (a->wr_launch != C.id) { a->wr_launch = C.id; a->wr_count = 0; a->wr_max = -1; }
            if (c->writer == -1) { a->wr_count++; if (ei > a->wr_max) a->wr_max = ei; }
            c->writer = t;
        }
    }
    if (rw & CUEMU_WRITE) memset(a->init + off, 1, esz);
    rec_add(a->id, ei, flags);
    return a->data + off;
}

static int kd_cmp(const void *x, const void *y)
{
    const kdiag_t *a = x, *b = y;
    return a->kind != b->kind ? a->kind - b->kind : a->line - b->line;
}

static void flush_kernel_diags(void)
{
    qsort(C.kd, (size_t)C.nkd, sizeof C.kd[0], kd_cmp);
    /* One out-of-bounds report per source line, even when it touches several arrays. */
    int merged[MAX_KDIAG] = {0};
    for (int i = 0; i < C.nkd; i++) {
        if (merged[i] || (C.kd[i].kind != KD_OOB_WRITE && C.kd[i].kind != KD_OOB_READ)) continue;
        for (int j = i + 1; j < C.nkd; j++)
            if ((C.kd[j].kind == KD_OOB_WRITE || C.kd[j].kind == KD_OOB_READ) && C.kd[j].line == C.kd[i].line &&
                C.kd[j].file == C.kd[i].file)
                merged[j] = i + 1;
    }
    for (int i = 0; i < C.nkd; i++) {
        if (merged[i]) continue;
        kdiag_t *d = &C.kd[i];
        alloc_t *a = d->alloc > 0 ? G.allocs[d->alloc - 1] : NULL;
        row_t r[8];
        int nr = 0;
        char title[400], hint[512], pos[160], cnt[32], thr[32], desc[400], idx[96];
        const char *code = "kernel";
        hint[0] = 0;
        fmt_int(d->count, cnt, sizeof cnt);
        fmt_int(d->threads, thr, sizeof thr);
        thread_pos(d->first_thread, pos, sizeof pos);
        if (d->min_index == d->max_index) snprintf(idx, sizeof idx, "%lld", d->min_index);
        else snprintf(idx, sizeof idx, "%lld … %lld", d->min_index, d->max_index);

        ROW(r[nr], "kernel", "`%s`, launched at line %d", C.kernel, C.line); nr++;
        for (int j = i; j < C.nkd && nr < 5; j++) {
            if (j != i && merged[j] != i + 1) continue;
            alloc_t *m = C.kd[j].alloc > 0 ? G.allocs[C.kd[j].alloc - 1] : NULL;
            if (!m) continue;
            alloc_desc(m, desc, sizeof desc);
            if (strcmp(m->name, C.kd[j].name) == 0) ROW(r[nr], "memory", "%s", desc);
            else ROW(r[nr], "memory", "`%s` is the device array %s", C.kd[j].name, desc);
            nr++;
        }
        long long elems = a && a->elem_size ? (long long)(a->size / (size_t)a->elem_size) : 0;
        int sev = SEV_ERROR;

        switch (d->kind) {
        case KD_OOB_WRITE:
        case KD_OOB_READ: {
            int w = d->kind == KD_OOB_WRITE;
            code = w ? "out-of-bounds-write" : "out-of-bounds-read";
            char others[200] = "";
            for (int j = i + 1; j < C.nkd; j++) {
                if (merged[j] != i + 1) continue;
                size_t n = strlen(others);
                snprintf(others + n, sizeof others - n, "%s`%s`", n ? ", " : "", C.kd[j].name);
            }
            snprintf(title, sizeof title, "kernel `%s` %s %s `%s`%s%s%s", C.kernel, w ? "wrote" : "read",
                     d->max_index < 0 ? "before the start of" : "past the end of", d->name,
                     others[0] ? " (and read " : "", others, others[0] ? " out of bounds)" : "");
            ROW(r[nr], "index", "%s, but valid indexes are 0 … %lld (%s bad %s from %s GPU threads)", idx,
                elems - 1, cnt, w ? "writes" : "reads", thr); nr++;
            ROW(r[nr], "first", "%s", pos); nr++;
            if (d->min_index >= elems && C.total > elems) {
                char tot[32], el[32], extra[32];
                fmt_int(C.total, tot, sizeof tot);
                fmt_int(elems, el, sizeof el);
                fmt_int(C.total - elems, extra, sizeof extra);
                snprintf(hint, sizeof hint,
                         "This launch made %s GPU threads for %s elements, so %s threads have no element. "
                         "Guard the work with a bounds check: if (i < n) { ... }", tot, el, extra);
            } else if (d->max_index < 0) {
                snprintf(hint, sizeof hint, "The index is negative. Check how i is computed, and guard neighbours such as i - 1.");
            } else {
                snprintf(hint, sizeof hint, "Check the index computation (i = blockIdx.x * blockDim.x + threadIdx.x) "
                                            "and guard it with if (i < n).");
            }
            break;
        }
        case KD_UNINIT_READ: {
            code = "uninitialized-read";
            sev = SEV_WARNING;
            snprintf(title, sizeof title, "kernel `%s` read `%s` before anything was written to it", C.kernel, d->name);
            ROW(r[nr], "index", "%s (%s reads from %s GPU threads)", idx, cnt, thr); nr++;
            ROW(r[nr], "first", "%s", pos); nr++;
            int never = a != NULL;
            for (size_t k = 0; a && k < a->size; k++)
                if (a->init[k]) { never = 0; break; }
            if (never)
                snprintf(hint, sizeof hint, "Nothing was ever copied into `%s`. Device memory starts out as garbage. "
                                            "Did you forget cudaMemcpy(%s, host_array, bytes, cudaMemcpyHostToDevice)?",
                         a->name, a->name);
            else
                snprintf(hint, sizeof hint, "Device memory starts out as garbage. Only part of `%s` was initialized; "
                                            "check the byte count of the cudaMemcpy that fills it.", a ? a->name : d->name);
            break;
        }
        case KD_RACE_WRITE:
        case KD_RACE_READ: {
            code = "data-race";
            char other[160];
            thread_pos(d->other_thread, other, sizeof other);
            if (d->kind == KD_RACE_WRITE)
                snprintf(title, sizeof title, "data race: several GPU threads write the same element of `%s`", d->name);
            else
                snprintf(title, sizeof title, "data race: GPU threads read elements of `%s` that other threads write", d->name);
            ROW(r[nr], "index", "%s (%s conflicting accesses from %s GPU threads)", idx, cnt, thr); nr++;
            ROW(r[nr], "threads", "%s … and %s", pos, other); nr++;
            snprintf(hint, sizeof hint, "%s",
                     d->kind == KD_RACE_WRITE
                         ? "On a GPU these threads run at the same time, so their updates overwrite each other and "
                           "the result is unpredictable (cuemu ran them one by one, so your output may look right). "
                           "Give each thread its own output element (out[i]), or use atomicAdd(&sum[0], value), "
                           "which updates memory in one indivisible step. See examples/11_atomic_sum.cu."
                         : "The value a thread reads depends on whether its neighbour already ran, which varies from "
                           "run to run on a GPU. Read from one array and write the results into a separate one.");
            break;
        }
        case KD_HOST_MEMORY: {
            code = "host-memory-in-kernel";
            int stack = d->addr >= G.stack_lo && d->addr < G.stack_hi;
            int pin = find_pinned(d->addr) != NULL;
            snprintf(title, sizeof title, "kernel `%s` used host memory through `%s`", C.kernel, d->name);
            ROW(r[nr], "address", "%p is %s, not device memory", (void *)d->addr,
                pin ? "pinned host memory from cudaMallocHost" : stack ? "a local array on the host stack" : "host memory (e.g. from malloc)"); nr++;
            ROW(r[nr], "first", "%s", pos); nr++;
            snprintf(hint, sizeof hint,
                     "GPU threads can only reach device memory. Allocate with cudaMalloc, copy the data with "
                     "cudaMemcpy(..., cudaMemcpyHostToDevice) and pass the device pointer to the kernel. "
                     "The device is now in an error state, so later CUDA calls fail (just like on a real GPU).");
            break;
        }
        case KD_NULL:
            code = "null-pointer";
            snprintf(title, sizeof title, "kernel `%s` dereferenced the NULL pointer `%s`", C.kernel, d->name);
            ROW(r[nr], "first", "%s", pos); nr++;
            snprintf(hint, sizeof hint, "Was cudaMalloc called for this pointer, and did it succeed?");
            break;
        case KD_USE_AFTER_FREE:
            code = "use-after-free";
            snprintf(title, sizeof title, "kernel `%s` used `%s` after cudaFree", C.kernel, d->name);
            ROW(r[nr], "freed", "at line %d", a ? a->free_line : 0); nr++;
            snprintf(hint, sizeof hint, "Free device memory only after the last kernel and copy that use it.");
            break;
        case KD_NOT_ALLOC:
            code = "bad-device-pointer";
            snprintf(title, sizeof title, "kernel `%s` used `%s`, which points between device allocations", C.kernel, d->name);
            ROW(r[nr], "first", "%s", pos); nr++;
            snprintf(hint, sizeof hint, "Pointer arithmetic went outside the allocation. Index from the pointer cudaMalloc returned.");
            break;
        }
        report(sev, code, d->file, d->line, d->alloc, title, r, nr, hint[0] ? hint : NULL);
    }

    /* Elements the launch never reached. */
    for (int i = 0; i < G.nallocs; i++) {
        alloc_t *a = G.allocs[i];
        if (a->freed || a->wr_launch != C.id || !a->elem_size) continue;
        long long elems = (long long)(a->size / (size_t)a->elem_size);
        if (a->wr_count <= 0 || a->wr_count >= elems || a->wr_max + 1 != a->wr_count || a->wr_count != C.total) continue;
        char title[320], tot[32], el[32], hint[400];
        fmt_int(C.total, tot, sizeof tot);
        fmt_int(elems, el, sizeof el);
        row_t r[2];
        ROW(r[0], "written", "elements 0 … %lld; elements %lld … %lld were never computed", a->wr_max, a->wr_max + 1, elems - 1);
        ROW(r[1], "threads", "the launch made %s GPU threads for %s elements", tot, el);
        snprintf(title, sizeof title, "kernel `%s` only covered part of `%s`", C.kernel, a->name);
        snprintf(hint, sizeof hint, "The block count was rounded down. Round it up: "
                                    "numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock.");
        report(SEV_WARNING, "partial-coverage", C.file, C.line, a->id, title, r, 2, hint);
    }
}

void cuemu_launch_end(cuemu_launch *L)
{
    if (L->ok) {
        if (L->next > 0) finish_thread();
        cuemu_tls = (cuemu_tls_state){0};
        G.threads_run += C.total;
    }
    flush_kernel_diags();

    double dur = 0, t0 = max_d(G.host_ms, G.busy_ms);
    if (L->ok) {
        long long rounds = (C.total + MODEL_CORES - 1) / MODEL_CORES;
        dur = MODEL_LAUNCH_MS + (double)rounds * (double)C.max_ops * MODEL_GPU_NS_PER_OP / 1e6;
        G.busy_ms = t0 + dur;
        G.kernel_ms += dur;
    }
    G.host_ms += MODEL_API_MS;

    if (ev_open("launch", C.file, C.line)) {
        sb_printf(&G.events, ",\"id\":%d,\"kernel\":", C.id);
        sb_json_str(&G.events, C.kernel);
        sb_printf(&G.events, ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"ok\":%s,\"reason\":", C.grid.x, C.grid.y,
                  C.grid.z, C.block.x, C.block.y, C.block.z, L->ok ? "true" : "false");
        sb_json_str(&G.events, C.reason);
        sb_printf(&G.events, ",\"threads\":%lld,\"tHost\":%g,\"t0\":%g,\"t1\":%g,\"opsMax\":%lld,\"opsTotal\":%lld,"
                             "\"cpuMs\":%g,\"recLimit\":%d,\"kfirst\":%d,\"klast\":%d,\"recDropped\":%lld,\"acc\":[",
                  C.total, C.t_host, t0, t0 + dur, C.max_ops, C.total_ops,
                  (double)C.total_ops * MODEL_CPU_NS_PER_OP / 1e6, REC_MAX_THREAD_ID, C.kfirst, C.klast, C.rec_dropped);
        for (size_t i = 0; i < C.nrec; i++)
            sb_printf(&G.events, "%s%d,%d,%lld,%d", i ? "," : "", C.rec[i].thread, C.rec[i].alloc, C.rec[i].index,
                      C.rec[i].flags);
        sb_puts(&G.events, "],\"order\":[");
        for (size_t i = 0; i < C.norder; i++) sb_printf(&G.events, "%s%lld", i ? "," : "", C.order[i]);
        sb_puts(&G.events, "],\"snaps\":[");
        int first = 1;
        for (int i = 0; i < G.nallocs; i++) {
            alloc_t *a = G.allocs[i];
            if (a->touched_launch != C.id || a->freed) continue;
            if (!first) sb_puts(&G.events, ",");
            first = 0;
            ev_snap(a);
        }
        sb_puts(&G.events, "]");
        ev_close();
    }
    C.active = 0;
    C.failed = 0;
    api_leave();
    if (G.strict && G.nerrors) {
        fprintf(stderr, "\n%scuemu: stopping at the first error (CUEMU_STRICT is set)%s\n", DIM, RESET);
        exit(1);
    }
}

/* ---- Crash handling ---------------------------------------------------------------- */

static int symbolize(uintptr_t pc, char *file, size_t flen, int *line)
{
#ifdef __APPLE__
    if (!G.exe[0] || !pc) return 0;
    const struct mach_header *mh = _dyld_get_image_header(0);
    char load[32], addr[32];
    snprintf(load, sizeof load, "0x%lx", (unsigned long)(uintptr_t)mh);
    snprintf(addr, sizeof addr, "0x%lx", (unsigned long)pc);
    int fds[2];
    if (pipe(fds) != 0) return 0;
    pid_t pid = fork();
    if (pid < 0) return 0;
    if (pid == 0) {
        dup2(fds[1], 1);
        int dn = open("/dev/null", O_WRONLY);
        if (dn >= 0) dup2(dn, 2);
        close(fds[0]);
        execl("/usr/bin/atos", "atos", "-o", G.exe, "-l", load, addr, (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    char out[1024];
    size_t n = 0;
    ssize_t k;
    while (n + 1 < sizeof out && (k = read(fds[0], out + n, sizeof out - 1 - n)) > 0) n += (size_t)k;
    out[n] = 0;
    close(fds[0]);
    waitpid(pid, NULL, 0);
    char *open_paren = strrchr(out, '(');
    char *colon = open_paren ? strrchr(open_paren, ':') : NULL;
    if (!open_paren || !colon) return 0;
    *colon = 0;
    const char *base = open_paren + 1;
    *line = atoi(colon + 1);
    snprintf(file, flen, "%s", base);
    for (int i = 0; i < G.nsrcs; i++) {
        const char *s = strrchr(G.srcs[i], '/');
        if (strcmp(s ? s + 1 : G.srcs[i], base) == 0) { snprintf(file, flen, "%s", G.srcs[i]); break; }
    }
    return *line > 0;
#else
    (void)pc; (void)file; (void)flen; (void)line;
    return 0;
#endif
}

static void crash_handler(int sig, siginfo_t *si, void *uctx)
{
    static volatile sig_atomic_t busy;
    if (busy) _exit(128 + sig);
    busy = 1;
    uintptr_t addr = (uintptr_t)si->si_addr, pc = 0;
#if defined(__APPLE__) && defined(__arm64__)
    pc = (uintptr_t)((ucontext_t *)uctx)->uc_mcontext->__ss.__pc;
#elif defined(__APPLE__) && defined(__x86_64__)
    pc = (uintptr_t)((ucontext_t *)uctx)->uc_mcontext->__ss.__rip;
#else
    (void)uctx;
#endif
    static char file[PATH_MAX];
    int line = 0;
    G.host_ms += now_real_ms() - G.last_real;
    int have = symbolize(pc, file, sizeof file, &line);
    G.last_real = now_real_ms();
    const char *f = have ? file : G.in_api ? G.api_file : NULL;
    if (!have && G.in_api) line = G.api_line;

    char title[400], desc[400], hint[512];
    row_t r[2];
    int nr = 0;
    const char *code = "crash";
    alloc_t *a = in_region(addr) ? find_alloc(addr) : NULL;
    if (!a && in_region(addr)) a = nearest_alloc(addr);
    hint[0] = 0;

    if (sig == SIGFPE) {
        code = "arithmetic-error";
        snprintf(title, sizeof title, "arithmetic error (integer division by zero?)%s%s%s",
                 cuemu_tls.in_kernel ? " in kernel `" : "", cuemu_tls.in_kernel ? C.kernel : "", cuemu_tls.in_kernel ? "`" : "");
    } else if (in_region(addr) && cuemu_tls.in_kernel) {
        code = "untracked-device-access";
        snprintf(title, sizeof title, "kernel `%s` reached device memory in a way cuemu cannot follow", C.kernel);
        snprintf(hint, sizeof hint, "cuemu checks accesses written as p[i] and *p. Rewrite other pointer tricks "
                                    "(p->field, *(p + i), memcpy) as p[i] or p[i].field.");
    } else if (in_region(addr) && G.in_api) {
        snprintf(title, sizeof title, "crash inside %s", G.in_api);
        snprintf(hint, sizeof hint, "%s", strcmp(G.in_api, "cudaMalloc") == 0
            ? "The first argument must be the address of your pointer: cudaMalloc(&d_a, bytes)."
            : "A host argument points at device memory. Check the order of dst and src.");
    } else if (in_region(addr)) {
        code = "device-pointer-on-host";
        snprintf(title, sizeof title, "host code dereferenced a device pointer");
        if (a) { alloc_desc(a, desc, sizeof desc); ROW(r[nr], "memory", "%s", desc); nr++; }
        ROW(r[nr], "address", "%p is in GPU memory, which the CPU cannot read or write", (void *)addr); nr++;
        snprintf(hint, sizeof hint, "Device pointers from cudaMalloc only work inside kernels and in cudaMemcpy. "
                                    "Copy results back first: cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "
                                    "then read the host array.");
    } else if (G.in_api) {
        snprintf(title, sizeof title, "crash inside %s: an argument is not valid host memory", G.in_api);
        snprintf(hint, sizeof hint, "%s", strcmp(G.in_api, "cudaMalloc") == 0
            ? "Pass the address of your pointer: cudaMalloc(&d_a, bytes)."
            : "Check that the host buffer is allocated and large enough for the byte count.");
    } else if (addr < 65536) {
        code = "null-pointer";
        snprintf(title, sizeof title, "segmentation fault: NULL pointer dereference%s%s%s",
                 cuemu_tls.in_kernel ? " in kernel `" : "", cuemu_tls.in_kernel ? C.kernel : "", cuemu_tls.in_kernel ? "`" : "");
        snprintf(hint, sizeof hint, "A pointer was NULL. Check the results of malloc and cudaMalloc.");
    } else {
        snprintf(title, sizeof title, "segmentation fault at address %p", (void *)addr);
    }
    if (!have) { ROW(r[nr], "where", "run under lldb for the exact line"); nr++; }
    report(SEV_ERROR, code, f, line, a ? a->id : -1, title, r, nr, hint[0] ? hint : NULL);
    if (ev_open("crash", f, line)) {
        sb_puts(&G.events, ",\"title\":");
        sb_json_str(&G.events, title);
        sb_printf(&G.events, ",\"t0\":%g,\"t1\":%g", G.host_ms, G.host_ms);
        ev_close();
    }
    G.crashed = 1;
    cuemu_tls.in_kernel = 0;
    finalize();
    /* Exit instead of re-raising: the explanation above beats a bare "Segmentation fault". */
    fflush(NULL);
    _exit(128 + sig);
}

/* ---- Exit: leaks, summary, trace ------------------------------------------------------ */

static void write_trace(void)
{
    if (!G.tracing) return;
    const char *out = getenv("CUEMU_TRACE_OUT");
    if (out && *out) snprintf(G.trace_path, sizeof G.trace_path, "%s", out);
    else snprintf(G.trace_path, sizeof G.trace_path, "%s.cuemu.html", G.exe[0] ? G.exe : "cuemu");

    const char *viewer = getenv("CUEMU_VIEWER");
    if (!viewer || !*viewer) viewer = CUEMU_DEFAULT_VIEWER;
    sbuf tpl = {0};
    FILE *vf = *viewer ? fopen(viewer, "rb") : NULL;
    if (vf) {
        char chunk[8192];
        size_t k;
        while ((k = fread(chunk, 1, sizeof chunk, vf)) > 0) sb_putn(&tpl, chunk, k);
        fclose(vf);
    }
    static const char marker[] = "\"__CUEMU_TRACE_JSON__\"";
    size_t plen = strlen(G.trace_path);
    int want_json = plen > 5 && strcmp(G.trace_path + plen - 5, ".json") == 0;
    char *at = tpl.p && !want_json ? strstr(tpl.p, marker) : NULL;

    sbuf j = {0};
    const char *prog = strrchr(G.exe, '/');
    char cwd[PATH_MAX];
    sb_puts(&j, "{\"version\":1,\"program\":");
    sb_json_str(&j, prog ? prog + 1 : G.exe);
    sb_puts(&j, ",\"cwd\":");
    sb_json_str(&j, getcwd(cwd, sizeof cwd) ? cwd : "");
    sb_printf(&j, ",\"status\":\"%s\",\"order\":\"%s\",\"seed\":%llu", G.crashed ? "crashed" : "ok",
              G.shuffle ? "shuffle" : "sequential", (unsigned long long)G.seed);
    sb_printf(&j, ",\"model\":{\"cores\":%d,\"gpuNsPerOp\":%g,\"cpuNsPerOp\":%g,\"launchMs\":%g,\"contextInitMs\":%g,"
                  "\"pageableGBps\":%g,\"pinnedGBps\":%g,\"latencyMs\":%g,\"maxThreadsPerBlock\":%d,\"deviceBytes\":%zu}",
              MODEL_CORES, MODEL_GPU_NS_PER_OP, MODEL_CPU_NS_PER_OP, MODEL_LAUNCH_MS, MODEL_CONTEXT_INIT_MS,
              MODEL_PCIE_PAGEABLE_GBPS, MODEL_PCIE_PINNED_GBPS, MODEL_PCIE_LATENCY_MS, MAX_THREADS_PER_BLOCK, G.limit);
    sb_printf(&j, ",\"totals\":{\"hostMs\":%g,\"endMs\":%g,\"kernelMs\":%g,\"h2dMs\":%g,\"d2hMs\":%g,\"waitMs\":%g,"
                  "\"launches\":%d,\"threads\":%lld,\"peakBytes\":%zu,\"errors\":%d,\"warnings\":%d,\"eventsDropped\":%d}",
              G.host_ms, max_d(G.host_ms, G.busy_ms), G.kernel_ms, G.h2d_ms, G.d2h_ms, G.wait_ms, G.nlaunches,
              G.threads_run, G.peak, G.nerrors, G.nwarnings, G.events_dropped);
    sb_puts(&j, ",\"events\":[");
    if (G.events.p) sb_putn(&j, G.events.p, G.events.n);
    sb_puts(&j, "],\"allocs\":[");
    for (int i = 0; i < G.nallocs; i++) {
        alloc_t *a = G.allocs[i];
        sb_printf(&j, "%s{\"id\":%d,\"name\":", i ? "," : "", a->id);
        sb_json_str(&j, a->name);
        sb_printf(&j, ",\"bytes\":%zu,\"elemSize\":%d,\"type\":%d,\"typeName\":\"%s\",\"line\":%d,\"freeLine\":%d}",
                  a->size, a->elem_size, a->type, type_name(a->type), a->line, a->freed ? a->free_line : 0);
    }
    sb_puts(&j, "],\"diagnostics\":[");
    if (G.diags.p) sb_putn(&j, G.diags.p, G.diags.n);
    sb_puts(&j, "],\"sources\":{");
    for (int i = 0; i < G.nsrcs; i++) {
        srcfile_t *s = source_get(G.srcs[i]);
        if (i) sb_puts(&j, ",");
        sb_json_str(&j, G.srcs[i]);
        sb_puts(&j, ":");
        if (s) sb_json_str(&j, s->text); else sb_puts(&j, "null");
    }
    sb_puts(&j, "}}");

    FILE *f = fopen(G.trace_path, "wb");
    if (!f) { G.trace_path[0] = 0; free(j.p); free(tpl.p); return; }
    if (at) {
        fwrite(tpl.p, 1, (size_t)(at - tpl.p), f);
        fwrite(j.p, 1, j.n, f);
        fputs(at + sizeof marker - 1, f);
    } else {
        /* No viewer template found: write plain JSON the viewer can open. */
        size_t n = strlen(G.trace_path);
        if (n > 5 && strcmp(G.trace_path + n - 5, ".html") == 0) {
            fclose(f);
            remove(G.trace_path);
            strcpy(G.trace_path + n - 5, ".json");
            f = fopen(G.trace_path, "wb");
            if (!f) { G.trace_path[0] = 0; free(j.p); free(tpl.p); return; }
        }
        fwrite(j.p, 1, j.n, f);
    }
    fclose(f);
    free(j.p);
    free(tpl.p);
}

static void finalize(void)
{
    if (!G.initialized || G.finalized) return;
    G.finalized = 1;
    fflush(stdout);
    double now = now_real_ms();
    G.host_ms += now - G.last_real;

    if (!G.crashed && !G.sticky) {
        char names[400] = "";
        int leaked = 0;
        size_t leaked_bytes = 0;
        for (int i = 0; i < G.nallocs; i++) {
            alloc_t *a = G.allocs[i];
            if (a->freed) continue;
            leaked++;
            leaked_bytes += a->size;
            size_t n = strlen(names);
            if (n < sizeof names - 80) snprintf(names + n, sizeof names - n, "%s`%s` (line %d)", n ? ", " : "", a->name, a->line);
        }
        if (leaked) {
            char title[200], lb[32];
            fmt_bytes(leaked_bytes, lb, sizeof lb);
            row_t r[1];
            ROW(r[0], "memory", "%s", names);
            snprintf(title, sizeof title, "%d device allocation%s (%s) %s never freed", leaked, leaked == 1 ? "" : "s", lb,
                     leaked == 1 ? "was" : "were");
            report(SEV_WARNING, "memory-leak", NULL, 0, -1, title, r, 1,
                   "Call cudaFree for every cudaMalloc once you are done with the memory.");
        }
        if (G.launch_fail_line) {
            char title[200];
            row_t r[1];
            ROW(r[0], "error", "%s (\"%s\")", cudaGetErrorName(cudaErrorInvalidConfiguration),
                cudaGetErrorString(cudaErrorInvalidConfiguration));
            snprintf(title, sizeof title, "a kernel launch failed and the program never checked for errors");
            report(SEV_WARNING, "unchecked-error", G.launch_fail_file, G.launch_fail_line, -1, title, r, 1,
                   "Kernel launches report failures silently. After a launch, check: "
                   "if (cudaGetLastError() != cudaSuccess) { ... }");
        }
    }

    write_trace();

    char tot[32], thr[32];
    fmt_bytes(G.peak, tot, sizeof tot);
    fmt_int(G.threads_run, thr, sizeof thr);
    double end = max_d(G.host_ms, G.busy_ms);
    fprintf(stderr, "\n%s── cuemu run summary %s─────────────────────────────────────────%s\n", BOLD, DIM, RESET);
    fprintf(stderr, "  %-10s %d launch%s, %s GPU threads\n", "kernels", G.nlaunches, G.nlaunches == 1 ? "" : "es", thr);
    fprintf(stderr, "  %-10s peak %s of device memory in %d allocation%s\n", "memory", tot, G.nallocs, G.nallocs == 1 ? "" : "s");
    fprintf(stderr, "  %-10s %.2f ms simulated: context %.0f · host→device %.3f · kernels %.3f · device→host %.3f\n",
            "time", end, MODEL_CONTEXT_INIT_MS, G.h2d_ms, G.kernel_ms, G.d2h_ms);
    fprintf(stderr, "  %-10s %s%d error%s%s, %s%d warning%s%s\n", "problems", G.nerrors ? RED : "", G.nerrors,
            G.nerrors == 1 ? "" : "s", RESET, G.nwarnings ? YELLOW : "", G.nwarnings, G.nwarnings == 1 ? "" : "s", RESET);
    if (G.trace_path[0]) fprintf(stderr, "  %-10s %sopen %s%s\n", "visualize", CYAN, rel_path(G.trace_path), RESET);
    fprintf(stderr, "\n");
}

static void at_exit(void)
{
    finalize();
    if (G.nerrors && !getenv("CUEMU_KEEP_EXIT_CODE")) {
        fflush(NULL);
        _exit(1);
    }
}
