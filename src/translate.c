/*
 * cuemu-translate — turns a CUDA C source file (.cu) into plain C for cuemu.
 *
 *   kernel<<<numBlocks, threadsPerBlock>>>(args);   ->  a loop over GPU threads
 *   a[i], *p, p->x   inside __global__/__device__   ->  CUEMU_AT(...) checked accesses
 *   dim3 b(16, 16);                                 ->  dim3 b = CUEMU_DIM3(16, 16);
 *
 * It also rejects things nvcc would reject (calling host functions from a
 * kernel, calling a kernel without <<< >>>, threadIdx in host code) with
 * messages aimed at students. Line numbers are preserved, so compiler
 * errors point at the original .cu file.
 *
 * usage: cuemu-translate input.cu [-o output.c]
 */
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { T_WS, T_COMMENT, T_PP, T_IDENT, T_NUM, T_STR, T_CHAR, T_PUNCT };

typedef struct {
    int k;
    const char *s;
    int n, line, col;
} tok;

enum { FN_GLOBAL = 1, FN_DEVICE = 2, FN_HOST = 4 };

typedef struct {
    char name[128];
    int flags, has_body, name_si, open_si, close_si, start_si;
} func;

static const char *path;
static char abspath[PATH_MAX];
static char *src;
static size_t srclen;
static tok *T;
static int nt;
static int *S;        /* indices of significant tokens (preprocessor lines included as barriers) */
static int ns;
static int *mt;       /* matching bracket for each significant token, or -1 */
static int *region;   /* 0 file scope, 1 host function body, 2 device/kernel body */
static int *owner;    /* function owning each significant token, or -1 */
static func *F;
static int nf, capf;
static char typedefs[256][64];
static int ntypedefs;
static int nerr, color;

typedef struct { char *p; size_t n, cap; } sbuf;
static sbuf out;

static void put(const char *s, size_t n)
{
    if (out.n + n + 1 > out.cap) {
        out.cap = (out.n + n + 1) * 2;
        out.p = realloc(out.p, out.cap);
    }
    memcpy(out.p + out.n, s, n);
    out.n += n;
    out.p[out.n] = 0;
}

static void puts_(const char *s) { put(s, strlen(s)); }

static void printf_(const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    put(buf, (size_t)(n < (int)sizeof buf ? n : (int)sizeof buf - 1));
}

/* ---- Tokenizer ------------------------------------------------------------- */

static void push(int k, size_t start, size_t end, int line, int col)
{
    static int cap;
    if (nt == cap) {
        cap = cap ? cap * 2 : 4096;
        T = realloc(T, sizeof *T * (size_t)cap);
    }
    T[nt++] = (tok){ k, src + start, (int)(end - start), line, col };
}

static void tokenize(void)
{
    static const char *puncts[] = {
        "<<<", ">>>", "<<=", ">>=", "...", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=",
        "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##",
    };
    size_t i = 0, line_start = 0;
    int line = 1, at_bol = 1;
    while (i < srclen) {
        size_t st = i;
        int tl = line, tc = (int)(i - line_start) + 1;
        char c = src[i];
        int k;
        #define NEWLINE_AT(j) do { line++; line_start = (j) + 1; } while (0)
        if (isspace((unsigned char)c)) {
            while (i < srclen && isspace((unsigned char)src[i])) {
                if (src[i] == '\n') { NEWLINE_AT(i); at_bol = 1; }
                i++;
            }
            push(T_WS, st, i, tl, tc);
            continue;
        }
        if (c == '/' && i + 1 < srclen && src[i + 1] == '/') {
            while (i < srclen && src[i] != '\n') i++;
            push(T_COMMENT, st, i, tl, tc);
            continue;
        }
        if (c == '/' && i + 1 < srclen && src[i + 1] == '*') {
            i += 2;
            while (i + 1 < srclen && !(src[i] == '*' && src[i + 1] == '/')) {
                if (src[i] == '\n') NEWLINE_AT(i);
                i++;
            }
            i = i + 2 <= srclen ? i + 2 : srclen;
            push(T_COMMENT, st, i, tl, tc);
            continue;
        }
        if (c == '#' && at_bol) {
            while (i < srclen && src[i] != '\n') {
                if (src[i] == '\\' && i + 1 < srclen && src[i + 1] == '\n') { NEWLINE_AT(i + 1); i += 2; continue; }
                i++;
            }
            push(T_PP, st, i, tl, tc);
            continue;
        }
        at_bol = 0;
        if (isalpha((unsigned char)c) || c == '_' || c == '$') {
            while (i < srclen && (isalnum((unsigned char)src[i]) || src[i] == '_' || src[i] == '$')) i++;
            k = T_IDENT;
        } else if (isdigit((unsigned char)c) || (c == '.' && i + 1 < srclen && isdigit((unsigned char)src[i + 1]))) {
            while (i < srclen) {
                char d = src[i];
                if ((d == '+' || d == '-') && strchr("eEpP", src[i - 1])) { i++; continue; }
                if (!isalnum((unsigned char)d) && d != '.' && d != '_' && d != '\'') break;
                i++;
            }
            k = T_NUM;
        } else if (c == '"' || c == '\'') {
            i++;
            while (i < srclen && src[i] != c && src[i] != '\n') i += src[i] == '\\' ? 2 : 1;
            if (i < srclen && src[i] == c) i++;
            k = c == '"' ? T_STR : T_CHAR;
        } else {
            size_t len = 1;
            for (size_t p = 0; p < sizeof puncts / sizeof *puncts; p++) {
                size_t pl = strlen(puncts[p]);
                if (i + pl <= srclen && memcmp(src + i, puncts[p], pl) == 0) { len = pl; break; }
            }
            i += len;
            k = T_PUNCT;
        }
        push(k, st, i, tl, tc);
    }
}

/* ---- Token helpers ------------------------------------------------------------ */

static tok *st(int si) { return si >= 0 && si < ns ? &T[S[si]] : NULL; }

static int is(int si, const char *p)
{
    tok *t = st(si);
    return t && (t->k == T_PUNCT || t->k == T_IDENT) && t->n == (int)strlen(p) && memcmp(t->s, p, (size_t)t->n) == 0;
}

static int kind(int si) { tok *t = st(si); return t ? t->k : -1; }

static int in_list(int si, const char *const *list)
{
    for (; *list; list++)
        if (is(si, *list)) return 1;
    return 0;
}

static const char *const keywords[] = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum",
    "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "_Bool", "_Complex", "_Generic", "_Static_assert", "_Alignof", "_Alignas",
    "_Thread_local", "__typeof__", "typeof", "__auto_type", "bool", "__attribute__", "__global__",
    "__device__", "__host__", "__shared__", NULL,
};

static const char *const type_words[] = {
    "void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "_Bool", "bool",
    "const", "volatile", "static", "register", "extern", "struct", "union", "enum", "auto", "__auto_type",
    "dim3", "size_t", "__typeof__", "typeof", NULL,
};

static const char *const expr_keywords[] = { "return", "case", "else", "do", "sizeof", "goto", NULL };

static int is_keyword(int si) { return in_list(si, keywords); }

static int is_type_name(int si)
{
    tok *t = st(si);
    if (!t || t->k != T_IDENT) return 0;
    if (in_list(si, type_words)) return 1;
    if (t->n > 2 && t->s[t->n - 2] == '_' && t->s[t->n - 1] == 't') return 1;
    for (int i = 0; i < ntypedefs; i++)
        if ((int)strlen(typedefs[i]) == t->n && memcmp(typedefs[i], t->s, (size_t)t->n) == 0) return 1;
    return 0;
}

static void tokname(int si, char *buf, size_t len)
{
    tok *t = st(si);
    snprintf(buf, len, "%.*s", t ? t->n : 0, t ? t->s : "");
}

static int find_func(int si)
{
    tok *t = st(si);
    if (!t || t->k != T_IDENT) return -1;
    for (int i = 0; i < nf; i++)
        if ((int)strlen(F[i].name) == t->n && memcmp(F[i].name, t->s, (size_t)t->n) == 0) return i;
    return -1;
}

static void error_at(int si, const char *fmt, ...)
{
    tok *t = st(si);
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "%s%s:%d:%d: %serror:%s %s", color ? "\033[1m" : "", path, t->line, t->col,
            color ? "\033[1;31m" : "", color ? "\033[0m\033[1m" : "", "");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "%s\n", color ? "\033[0m" : "");
    va_end(ap);
    const char *ls = t->s;
    while (ls > src && ls[-1] != '\n') ls--;
    const char *le = t->s;
    while (le < src + srclen && *le != '\n') le++;
    fprintf(stderr, "%5d | %.*s\n      | ", t->line, (int)(le - ls), ls);
    for (const char *p = ls; p < t->s; p++) fputc(*p == '\t' ? '\t' : ' ', stderr);
    fprintf(stderr, "%s^%s\n", color ? "\033[1;32m" : "", color ? "\033[0m" : "");
    nerr++;
}

static void note(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "      %s= hint:%s ", color ? "\033[1;36m" : "", color ? "\033[0m" : "");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

/* ---- Structure: brackets, functions, typedefs --------------------------------------- */

static void match_brackets(void)
{
    mt = malloc(sizeof(int) * (size_t)(ns + 1));
    int *stack = malloc(sizeof(int) * (size_t)(ns + 1)), sp = 0;
    for (int i = 0; i < ns; i++) {
        mt[i] = -1;
        if (is(i, "(") || is(i, "[") || is(i, "{")) stack[sp++] = i;
        else if (is(i, ")") || is(i, "]") || is(i, "}")) {
            if (sp) { int o = stack[--sp]; mt[o] = i; mt[i] = o; }
        }
    }
    free(stack);
}

static void add_func(int name_si, int start_si, int flags, int open_si)
{
    int idx = find_func(name_si);
    if (idx < 0) {
        if (nf == capf) { capf = capf ? capf * 2 : 64; F = realloc(F, sizeof *F * (size_t)capf); }
        idx = nf++;
        memset(&F[idx], 0, sizeof F[idx]);
        tokname(name_si, F[idx].name, sizeof F[idx].name);
        F[idx].open_si = F[idx].close_si = -1;
    }
    F[idx].flags |= flags;
    if (open_si >= 0) {
        F[idx].has_body = 1;
        F[idx].name_si = name_si;
        F[idx].start_si = start_si;
        F[idx].open_si = open_si;
        F[idx].close_si = mt[open_si];
    }
}

static void scan_file_scope(void)
{
    for (int si = 0; si < ns; si++) {
        if (kind(si) == T_PP) continue;
        if (is(si, "typedef")) {
            int last = -1, k = si + 1;
            for (; k < ns && !is(k, ";"); k++) {
                if ((is(k, "{") || is(k, "(") || is(k, "[")) && mt[k] > 0) { k = mt[k]; continue; }
                if (kind(k) == T_IDENT) last = k;
            }
            if (last >= 0 && ntypedefs < 256) tokname(last, typedefs[ntypedefs++], 64);
            si = k;
            continue;
        }
        if (is(si, "{") && mt[si] > 0) { si = mt[si]; continue; }
        if (kind(si) != T_IDENT || is_keyword(si) || !is(si + 1, "(") || mt[si + 1] < 0) continue;
        int close = mt[si + 1], after = close + 1;
        if (!is(after, "{") && !is(after, ";") && !is(after, ",")) continue;

        int flags = 0, returns_void = 0, star = 0, start = si;
        for (int b = si - 1; b >= 0; b--) {
            if (kind(b) == T_PP || is(b, ";") || is(b, "}") || is(b, "{")) break;
            start = b;
            if (is(b, "__global__")) flags |= FN_GLOBAL;
            else if (is(b, "__device__")) flags |= FN_DEVICE;
            else if (is(b, "__host__")) flags |= FN_HOST;
            else if (is(b, "void")) returns_void = 1;
            else if (is(b, "*")) star = 1;
        }
        if ((flags & FN_GLOBAL) && (!returns_void || star)) {
            char name[128];
            tokname(si, name, sizeof name);
            error_at(si, "kernel `%s` must return void", name);
            note("A __global__ function cannot return a value. Write results into a device array instead.");
        }
        if ((flags & FN_GLOBAL) && (flags & (FN_DEVICE | FN_HOST))) {
            error_at(si, "a function cannot be both __global__ and __device__/__host__");
        }
        add_func(si, start, flags, is(after, "{") ? after : -1);
        si = is(after, "{") ? mt[after] : close;
    }

    region = calloc((size_t)ns + 1, sizeof(int));
    owner = malloc(sizeof(int) * ((size_t)ns + 1));
    for (int i = 0; i <= ns; i++) owner[i] = -1;
    for (int f = 0; f < nf; f++) {
        if (!F[f].has_body || F[f].close_si < 0) continue;
        int r = F[f].flags & (FN_GLOBAL | FN_DEVICE) ? 2 : 1;
        for (int i = F[f].open_si; i <= F[f].close_si; i++) { region[i] = r; owner[i] = f; }
    }
}

/* ---- Context tests -------------------------------------------------------------------- */

static int prev_is_value_end(int p)
{
    /* Does token p end an operand (so a following * or & is binary)? */
    if (p < 0 || kind(p) == T_PP) return 0;
    if (kind(p) == T_NUM || kind(p) == T_STR || kind(p) == T_CHAR) return 1;
    if (kind(p) == T_IDENT) return !in_list(p, expr_keywords);
    if (is(p, ")") || is(p, "]")) return 1;
    if (is(p, "++") || is(p, "--")) return prev_is_value_end(p - 1);
    return 0;
}

static int inside_parens(int si)
{
    int depth = 0;
    for (int b = si - 1; b >= 0; b--) {
        if (is(b, ")") || is(b, "]")) depth++;
        else if (is(b, "(") || is(b, "[")) { if (depth == 0) return 1; depth--; }
        else if (depth == 0 && (is(b, ";") || is(b, "{") || is(b, "}") || kind(b) == T_PP)) return 0;
    }
    return 0;
}

/* Is the identifier at si being declared (float tmp[4]; float a, b[4]) rather than used? */
static int is_declarator(int si)
{
    int p = si - 1;
    if (kind(p) == T_IDENT && !in_list(p, expr_keywords)) return 1;
    if (is(p, "*")) {
        while (is(p, "*") || is(p, "const") || is(p, "restrict") || is(p, "volatile")) p--;
        return is_type_name(p);
    }
    if (is(p, ",") && !inside_parens(p)) {
        int b = p - 1;
        for (; b >= 0; b--) {
            if ((is(b, ")") || is(b, "]")) && mt[b] >= 0) { b = mt[b]; continue; }
            if (is(b, "}") && mt[b] > 0 && is(mt[b] - 1, "=")) { b = mt[b]; continue; }
            if (is(b, ";") || is(b, "{") || is(b, "}") || kind(b) == T_PP) break;
        }
        return is_type_name(b + 1);
    }
    return 0;
}

static int in_unevaluated(int si)
{
    static const char *const ops[] = { "sizeof", "_Alignof", "__typeof__", "typeof", NULL };
    if (in_list(si - 1, ops)) return 1;
    return is(si - 1, "(") && in_list(si - 2, ops);
}

static const char *access_mode(int first, int last)
{
    int prev = first - 1;
    if (is(prev, "&") && !prev_is_value_end(prev - 1)) return "CUEMU_ADDR";
    int prefix = (is(prev, "++") || is(prev, "--")) && !prev_is_value_end(prev - 1);
    int k = last + 1;
    for (;;) {
        if ((is(k, ".") || is(k, "->")) && kind(k + 1) == T_IDENT) { k += 2; continue; }
        if (is(k, "[") && mt[k] > 0) { k = mt[k] + 1; continue; }
        break;
    }
    static const char *const compound[] = { "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", NULL };
    if (prefix || in_list(k, compound) || is(k, "++") || is(k, "--")) return "CUEMU_RW";
    if (is(k, "=")) return "CUEMU_WRITE";
    return "CUEMU_READ";
}

/* ---- Emission ---------------------------------------------------------------------------- */

static void emit_tokens(int from_t, int to_t)
{
    for (int t = from_t; t < to_t; t++) put(T[t].s, (size_t)T[t].n);
}

static void trivia(int si) { emit_tokens(S[si] + 1, S[si + 1]); }

static void emit_range(int from, int to);

/* Everything strictly between the brackets at open and mt[open]. */
static void emit_inner(int open) { trivia(open); emit_range(open + 1, mt[open]); }

static int count_nl(const char *s, size_t n)
{
    int c = 0;
    for (size_t i = 0; i < n; i++) c += s[i] == '\n';
    return c;
}

/* After replacing significant tokens first..last, restore any newlines that were dropped. */
static int finish_rewrite(size_t out_start, int first, int last)
{
    const char *a = T[S[first]].s, *b = T[S[last]].s + T[S[last]].n;
    int missing = count_nl(a, (size_t)(b - a)) - count_nl(out.p + out_start, out.n - out_start);
    for (; missing > 0; missing--) puts_("\n");
    trivia(last);
    return last + 1;
}

static void put_tok(int si) { tok *t = st(si); put(t->s, (size_t)t->n); }

static int split_commas(int from, int to, int *starts, int max)
{
    int n = 0;
    if (from >= to) return 0;
    starts[n++] = from;
    for (int i = from; i < to; i++) {
        if ((is(i, "(") || is(i, "[") || is(i, "{")) && mt[i] > 0) { i = mt[i]; continue; }
        if (is(i, ",") && n < max) starts[n++] = i + 1;
    }
    return n;
}

static const char *const builtins[] = { "threadIdx", "blockIdx", "blockDim", "gridDim", NULL };

static const char *const host_only[] = {
    "malloc", "calloc", "realloc", "free", "fopen", "fclose", "fread", "fwrite", "fscanf", "scanf",
    "fgets", "gets", "getchar", "putchar", "puts", "exit", "abort", "system", "time", "clock", "rand",
    "srand", "sleep", "usleep", "fflush", "fprintf", "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemset",
    "cudaDeviceSynchronize", "cudaGetLastError", "cudaPeekAtLastError", "cudaMallocHost", "cudaFreeHost",
    NULL,
};

static int rewrite_launch(int i, int to)
{
    char name[128];
    tokname(i, name, sizeof name);
    int gt = -1;
    for (int k = i + 2; k < to; k++) {
        if ((is(k, "(") || is(k, "[") || is(k, "{")) && mt[k] > 0) { k = mt[k]; continue; }
        if (is(k, ">>>")) { gt = k; break; }
        if (is(k, ";")) break;
    }
    if (gt < 0) { error_at(i + 1, "missing `>>>` in the launch of kernel `%s`", name); return 0; }
    if (!is(gt + 1, "(") || mt[gt + 1] < 0) {
        error_at(gt, "expected `(` with the kernel arguments after `>>>`");
        return 0;
    }
    if (region[i] == 2) {
        error_at(i, "a kernel cannot launch another kernel");
        note("Launch kernels from host code (for example from main).");
    }
    int f = find_func(i);
    if (f >= 0 && !(F[f].flags & FN_GLOBAL)) {
        error_at(i, "`%s` is not a kernel, so it cannot be launched with <<< >>>", name);
        note("Mark the function __global__ to make it a kernel.");
    }

    int cfg[8], ncfg = split_commas(i + 2, gt, cfg, 8);
    if (ncfg != 2) {
        error_at(i + 1, ncfg > 2 ? "cuemu supports only <<<numBlocks, threadsPerBlock>>> (no stream or shared-memory size)"
                                 : "a launch needs two values: kernel<<<numBlocks, threadsPerBlock>>>");
        return 0;
    }
    int open = gt + 1, close = mt[open];
    int args[128], nargs = split_commas(open + 1, close, args, 128);

    size_t start = out.n;
    puts_("do { ");
    for (int a = 0; a < nargs; a++) {
        int end = a + 1 < nargs ? args[a + 1] - 1 : close;
        printf_("__auto_type _cuemu_a%d = (", a);
        emit_range(args[a], end);
        puts_("); ");
    }
    printf_("cuemu_launch _cuemu_L; if (cuemu_launch_begin(&_cuemu_L, \"%s\", CUEMU_TO_DIM3(", name);
    emit_range(cfg[0], cfg[1] - 1);
    puts_("), CUEMU_TO_DIM3(");
    emit_range(cfg[1], gt);
    printf_("), __FILE__, %d)) { while (cuemu_launch_next(&_cuemu_L)) %s(", st(i)->line, name);
    for (int a = 0; a < nargs; a++) printf_("%s_cuemu_a%d", a ? ", " : "", a);
    puts_("); } cuemu_launch_end(&_cuemu_L); } while (0)");
    return finish_rewrite(start, i, close);
}

static int rewrite_dim3_decl(int i, int to)
{
    /* dim3 a(16, 16), b;  ->  dim3 a = CUEMU_DIM3(16, 16), b = {1, 1, 1}; */
    size_t start = out.n;
    put_tok(i);
    trivia(i);
    int k = i + 1;
    while (k < to && kind(k) == T_IDENT) {
        put_tok(k);
        if (is(k + 1, "(") && mt[k + 1] > 0) {
            puts_(" = CUEMU_DIM3(");
            emit_inner(k + 1);
            puts_(")");
            k = mt[k + 1] + 1;
        } else if (is(k + 1, ",") || is(k + 1, ";")) {
            puts_(" = {1, 1, 1}");
            k = k + 1;
        } else if (is(k + 1, "=")) {
            int e = k + 2;
            while (e < to && !is(e, ",") && !is(e, ";")) {
                if ((is(e, "(") || is(e, "[") || is(e, "{")) && mt[e] > 0) e = mt[e];
                e++;
            }
            puts_(" = ");
            if (is(k + 2, "{")) emit_range(k + 2, e);
            else { puts_("CUEMU_TO_DIM3("); emit_range(k + 2, e); puts_(")"); }
            k = e;
        } else {
            break;
        }
        if (!is(k, ",")) break;
        puts_(", ");
        k++;
    }
    if (k == i + 1) return 0;
    /* k is the first token not consumed (usually ';'). */
    size_t keep = out.n;
    int r = finish_rewrite(start, i, k - 1);
    (void)keep;
    return r;
}

static int try_rewrite(int i, int to)
{
    tok *t = st(i);
    if (t->k == T_PP) return 0;
    int r = region[i];

    if (is(i, "__shared__")) {
        error_at(i, "shared memory (__shared__) is not supported by cuemu yet");
        return 0;
    }

    if (r && owner[i] >= 0 && F[owner[i]].open_si == i && (F[owner[i]].flags & FN_GLOBAL)) {
        func *f = &F[owner[i]];
        put_tok(i);
        printf_(" cuemu_kernel_enter(__builtin_frame_address(0), \"%s\", %d, %d);", f->name, st(f->start_si)->line,
                st(f->close_si)->line);
        trivia(i);
        return i + 1;
    }
    if (!r) return 0;

    if (t->k == T_IDENT && is(i + 1, "<<<")) return rewrite_launch(i, to);

    int after_member = is(i - 1, ".") || is(i - 1, "->");

    /* dim3 constructors */
    if (is(i, "dim3") && !after_member) {
        if (is(i + 1, "(")) {
            puts_("CUEMU_DIM3");
            trivia(i);
            return i + 1;
        }
        if (kind(i + 1) == T_IDENT && !is_keyword(i + 1) &&
            (is(i + 2, "(") || is(i + 2, ";") || is(i + 2, ",") || is(i + 2, "=")))
            return rewrite_dim3_decl(i, to);
    }

    if (t->k == T_IDENT && !after_member && !is_keyword(i)) {
        char name[128];
        tokname(i, name, sizeof name);
        int f = find_func(i);
        int call = is(i + 1, "(");

        if (r == 1) {
            if (in_list(i, builtins)) {
                error_at(i, "`%s` only exists inside kernels", name);
                note("threadIdx, blockIdx, blockDim and gridDim describe the GPU thread that is running. "
                     "Use them in __global__ or __device__ functions.");
            } else if (call && f >= 0 && (F[f].flags & FN_GLOBAL)) {
                error_at(i, "kernel `%s` must be launched, not called", name);
                note("Write %s<<<numBlocks, threadsPerBlock>>>(...) to run it on the GPU.", name);
            } else if (call && f >= 0 && (F[f].flags & FN_DEVICE) && !(F[f].flags & FN_HOST)) {
                error_at(i, "`%s` is a __device__ function and cannot be called from host code", name);
                note("Mark it `__host__ __device__` to use it on both sides.");
            }
        } else if (call && r == 2) {
            if (f >= 0 && (F[f].flags & FN_GLOBAL)) {
                error_at(i, "a kernel cannot call the kernel `%s`", name);
                note("Move the shared code into a __device__ function and call that.");
            } else if (f >= 0 && !(F[f].flags & FN_DEVICE)) {
                error_at(i, "`%s` is a host function and cannot be called from GPU code", name);
                note("Mark it `__device__` (or `__host__ __device__`) so it can run on the GPU.");
            } else if (f < 0 && in_list(i, host_only)) {
                error_at(i, "`%s` cannot be called from GPU code", name);
                note("Kernels run on the GPU. Memory management, files and random numbers belong in host code. "
                     "printf is allowed.");
            }
        }

        /* checked subscripts */
        if (r == 2 && is(i + 1, "[") && mt[i + 1] > 0 && mt[i + 1] < to && !is_declarator(i) && !in_unevaluated(i)) {
            int close = mt[i + 1];
            const char *mode = access_mode(i, close);
            size_t start = out.n;
            puts_("CUEMU_AT(");
            put_tok(i);
            puts_(", (");
            emit_inner(i + 1);
            printf_("), %s)", mode);
            return finish_rewrite(start, i, close);
        }

        /* p->field */
        if (r == 2 && is(i + 1, "->") && kind(i + 2) == T_IDENT && !in_unevaluated(i)) {
            const char *mode = access_mode(i, i);
            size_t start = out.n;
            printf_("CUEMU_AT(%s, 0, %s).", name, mode);
            return finish_rewrite(start, i, i + 1);
        }
    }

    /* *p and *(expr) */
    if (r == 2 && is(i, "*") && !prev_is_value_end(i - 1) && !(is(i - 1, ",") && !inside_parens(i - 1)) &&
        !in_unevaluated(i) && !is_type_name(i - 1)) {
        static const char *const blocked[] = { "[", "(", ".", "->", "++", "--", NULL };
        if (kind(i + 1) == T_IDENT && !is_keyword(i + 1) && !in_list(i + 2, blocked)) {
            const char *mode = access_mode(i, i + 1);
            size_t start = out.n;
            puts_("CUEMU_AT(");
            put_tok(i + 1);
            printf_(", 0, %s)", mode);
            return finish_rewrite(start, i, i + 1);
        }
        if (is(i + 1, "(") && mt[i + 1] > 0 && mt[i + 1] < to) {
            int close = mt[i + 1];
            int n = close + 1;
            if (kind(n) != T_IDENT && kind(n) != T_NUM && !in_list(n, blocked) && !is(n, "*") && !is(n, "&")) {
                const char *mode = access_mode(i, close);
                size_t start = out.n;
                puts_("CUEMU_AT((");
                emit_inner(i + 1);
                printf_("), 0, %s)", mode);
                return finish_rewrite(start, i, close);
            }
        }
    }

    if (r == 2 && is(i, "<<<")) error_at(i, "kernel launches are not allowed inside GPU code");
    return 0;
}

static void emit_range(int from, int to)
{
    int i = from;
    while (i < to) {
        int next = try_rewrite(i, to);
        if (next > i) { i = next; continue; }
        emit_tokens(S[i], S[i + 1]);
        i++;
    }
}

/* ---- Main ---------------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    const char *output = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output = argv[++i];
        else if (!path) path = argv[i];
        else { fprintf(stderr, "usage: cuemu-translate input.cu [-o output.c]\n"); return 2; }
    }
    if (!path) { fprintf(stderr, "usage: cuemu-translate input.cu [-o output.c]\n"); return 2; }
    color = isatty(2) && !getenv("NO_COLOR");

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    src = malloc((size_t)len + 1);
    srclen = fread(src, 1, (size_t)len, f);
    src[srclen] = 0;
    fclose(f);
    if (!realpath(path, abspath)) snprintf(abspath, sizeof abspath, "%s", path);

    tokenize();
    S = malloc(sizeof(int) * ((size_t)nt + 1));
    for (int t = 0; t < nt; t++)
        if (T[t].k != T_WS && T[t].k != T_COMMENT) S[ns++] = t;
    S[ns] = nt;
    match_brackets();
    scan_file_scope();

    puts_("/* Generated by cuemu-translate. Edit the .cu file instead. */\n#include \"cuemu.h\"\n#line 1 \"");
    for (const char *p = abspath; *p; p++) {
        if (*p == '"' || *p == '\\') puts_("\\");
        put(p, 1);
    }
    puts_("\"\n");
    emit_tokens(0, ns ? S[0] : nt);
    emit_range(0, ns);

    if (nerr) {
        fprintf(stderr, "cuemu-translate: %d error%s in %s\n", nerr, nerr == 1 ? "" : "s", path);
        return 1;
    }
    FILE *o = output ? fopen(output, "wb") : stdout;
    if (!o) { perror(output); return 1; }
    fwrite(out.p, 1, out.n, o);
    if (output) fclose(o);
    return 0;
}
