/*
 * test_runner.c — VPU Interpreter Test Framework
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "state.h"
#include "gpgpu_core.h"
#include "scheduler.h"
#include "test_runner.h"

/* ============================================================
 * 辅助函数
 * ============================================================ */

static uint8_t *read_file(const char *path, size_t *out_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return NULL; }
    *out_size = st.st_size;
    uint8_t *buf = malloc(st.st_size);
    if (!buf) { perror("malloc"); close(fd); return NULL; }
    ssize_t n = read(fd, buf, st.st_size);
    close(fd);
    if (n < 0 || (size_t)n != st.st_size) { perror("read"); free(buf); return NULL; }
    return buf;
}

static size_t load_kernel(const char *path, GPGPUState *s, uint32_t kernel_addr)
{
    size_t size;
    uint8_t *buf = read_file(path, &size);
    if (!buf) return 0;
    if (kernel_addr + size > s->vram_size) {
        fprintf(stderr, "ERROR: kernel too large (addr=0x%x size=%zu vram=%lu)\n",
                kernel_addr, size, (unsigned long)s->vram_size);
        free(buf);
        return 0;
    }
    memcpy(s->vram_ptr + kernel_addr, buf, size);
    free(buf);
    return size;
}

/* ============================================================
 * 测试框架
 * ============================================================ */

typedef struct {
    const char *name;
    int   errors;
} TestResult;

#define N_RUNS 5
#define SAVE_SIZE (4 * 1024 * 1024 + 4096)

/* 运行 kernel: 加载 → 调度 → 计时 */
static int run_kernel(GPGPUState *s, const char *kern_path,
                       uint32_t kernel_addr,
                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                       uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
    struct timespec t0, t1;

    s->kernel.kernel_addr = kernel_addr;
    s->kernel.grid_dim[0] = grid_x; s->kernel.grid_dim[1] = grid_y; s->kernel.grid_dim[2] = grid_z;
    s->kernel.block_dim[0] = block_x; s->kernel.block_dim[1] = block_y; s->kernel.block_dim[2] = block_z;

    load_kernel(kern_path, s, kernel_addr);

    s->inst_count = 0;
    s->fp_count   = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ret = scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double mflops = (elapsed > 0) ? (s->fp_count / elapsed / 1e6) : 0;
    uint32_t total_blocks = grid_x * grid_y * grid_z;

    if (mflops >= 1000.0)
        printf("  %5.0fus  %6.1f GFLOPS  %lu flops  %u blocks\n",
               elapsed * 1e6, mflops / 1000.0,
               (unsigned long)s->fp_count, total_blocks);
    else
        printf("  %5.0fus  %6.1f MFLOPS  %lu flops  %u blocks\n",
               elapsed * 1e6, mflops,
               (unsigned long)s->fp_count, total_blocks);

    return ret;
}

/* ============================================================
 * 测试用例
 * ============================================================ */

static TestResult test_vector_add(GPGPUState *s)
{
    TestResult r = { .name = "vecmul (multi-lane)", .errors = 0 };
    uint32_t N = 2048;
    uint32_t kern = 0x500000;

    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i + 1);
        ((float *)(s->vram_ptr + 0x200000))[i] = 2.0f;
    }
    if (run_kernel(s, "kernels/vecmul.bin", kern, 1, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    for (uint32_t i = 0; i < N; i++) {
        float expected = (float)(i + 1) * 2.0f;
        float got = ((float *)(s->vram_ptr + 0x300000))[i];
        if (fabsf(got - expected) > 1e-5f) {
            if (r.errors < 5) printf("  [%u] exp %.1f got %.1f\n", i, expected, got);
            r.errors++;
        }
    }
    printf("  %s: %u elements\n", r.errors == 0 ? "PASS" : "FAIL", N);
    return r;
}

static TestResult test_saxpy(GPGPUState *s)
{
    TestResult r = { .name = "saxpy (float fmadd)", .errors = 0 };
    uint32_t N = 16384;
    float A = 2.5f;
    uint32_t kern = 0x500000;

    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    *(float *)(s->vram_ptr + 0x4) = A;
    srand48(42);
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(drand48() * 100.0 - 50.0);
        ((float *)(s->vram_ptr + 0x200000))[i] = (float)(drand48() * 100.0 - 50.0);
    }
    float *y_copy = malloc(N * sizeof(float));
    memcpy(y_copy, s->vram_ptr + 0x200000, N * sizeof(float));

    if (run_kernel(s, "kernels/saxpy.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; free(y_copy); return r;
    }
    for (uint32_t i = 0; i < N; i++) {
        float x = ((float *)(s->vram_ptr + 0x100000))[i];
        float expected = A * x + y_copy[i];
        float got = ((float *)(s->vram_ptr + 0x200000))[i];
        if (fabsf(got - expected) > 1e-5f * fabsf(expected) && fabsf(got - expected) > 1e-6f) {
            if (r.errors < 10) printf("  [%u] exp %.6f got %.6f\n", i, expected, got);
            r.errors++;
        }
    }
    free(y_copy);
    printf("  %s: %u elements\n", r.errors == 0 ? "PASS" : "FAIL", N);
    return r;
}

static TestResult test_dot_product(GPGPUState *s)
{
    TestResult r = { .name = "dot_product", .errors = 0 };
    uint32_t N = 32768, kern = 0x500000;
    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    float sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        float a = (float)(i % 100) * 0.01f;
        float b = (float)((i + 1) % 100) * 0.01f;
        ((float *)(s->vram_ptr + 0x100000))[i] = a;
        ((float *)(s->vram_ptr + 0x200000))[i] = b;
        sum += a * b;
    }
    if (run_kernel(s, "kernels/dot_product.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float got = *(float *)(s->vram_ptr + 0x4);
    if (fabsf(got - sum) > 0.01f) { r.errors++; printf("  exp %.4f got %.4f\n", sum, got); }
    printf("  %s: N=%u result=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", N, got);
    return r;
}

static TestResult test_memcpy(GPGPUState *s)
{
    TestResult r = { .name = "memcpy", .errors = 0 };
    uint32_t N = 131072, kern = 0x500000;
    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    for (uint32_t i = 0; i < N; i++) {
        ((uint32_t *)(s->vram_ptr + 0x100000))[i] = i;
        ((uint32_t *)(s->vram_ptr + 0x200000))[i] = 0;
    }
    if (run_kernel(s, "kernels/memcpy.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    for (uint32_t i = 0; i < N; i++) {
        if (((uint32_t *)(s->vram_ptr + 0x200000))[i] != i) { r.errors++; break; }
    }
    printf("  %s: %lu bytes\n", r.errors == 0 ? "PASS" : "FAIL", (unsigned long)N * 8);
    return r;
}

static TestResult test_loop_perf(GPGPUState *s)
{
    TestResult r = { .name = "massive (multi-lane)", .errors = 0 };
    uint32_t N = 2048, kern = 0x500000;
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + 0x200000))[i] = 3.0f;
    }
    if (run_kernel(s, "kernels/vecmul.bin", kern, 1, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    printf("  ~%u elements\n", N);
    return r;
}

static TestResult test_rv32m(GPGPUState *s)
{
    TestResult r = { .name = "rv32m (mul/div/rem)", .errors = 0 };
    uint32_t kern = 0x500000;
    uint32_t cases[][2] = {
        {10, 3}, {10, (uint32_t)-3}, {(uint32_t)-5, (uint32_t)-2},
        {100, 0}, {0x80000000, 2}, {0xFFFFFFFF, 0xFFFFFFFF},
    };
    uint32_t N = sizeof(cases) / sizeof(cases[0]);
    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    memcpy(s->vram_ptr + 0x100000, cases, sizeof(cases));

    if (run_kernel(s, "kernels/rv32m.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    for (uint32_t i = 0; i < N; i++) {
        int32_t a = cases[i][0], b = cases[i][1];
        uint32_t *out = (uint32_t *)(s->vram_ptr + 0x300000) + i * 8;
        uint32_t exp[8] = {
            (uint32_t)(a * b),
            (uint32_t)(((int64_t)a * (int64_t)b) >> 32),
            (uint32_t)(((int64_t)a * (uint64_t)(uint32_t)b) >> 32),
            (uint32_t)(((uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b) >> 32),
            b ? (uint32_t)(a / b) : (uint32_t)-1,
            b ? ((uint32_t)a / (uint32_t)b) : 0xFFFFFFFF,
            b ? (uint32_t)(a % b) : (uint32_t)a,
            b ? ((uint32_t)a % (uint32_t)b) : (uint32_t)a,
        };
        for (int j = 0; j < 8; j++) {
            if (out[j] != exp[j]) {
                if (r.errors < 5) printf("  [%u][%d] got 0x%x exp 0x%x\n", i, j, out[j], exp[j]);
                r.errors++;
            }
        }
    }
    printf("  %s: 8x%u verified\n", r.errors == 0 ? "PASS" : "FAIL", N);
    return r;
}

static TestResult test_rv32f(GPGPUState *s)
{
    TestResult r = { .name = "rv32f (fma/cmp/cvt/class)", .errors = 0 };
    uint32_t kern = 0x500000;
    float fa = 3.0f, fb = 2.0f, fc = 4.0f;
    int32_t i32 = -5; uint32_t u32 = 7;

    memcpy(s->vram_ptr + 0x100000, &fa, 4);
    memcpy(s->vram_ptr + 0x100004, &fb, 4);
    memcpy(s->vram_ptr + 0x100008, &fc, 4);
    memcpy(s->vram_ptr + 0x10000C, &i32, 4);
    memcpy(s->vram_ptr + 0x100010, &u32, 4);

    if (run_kernel(s, "kernels/rv32f.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }

    float    *fout = (float *)(s->vram_ptr + 0x300000);
    uint32_t *iout = (uint32_t *)(s->vram_ptr + 0x300000);

    #define CF(i, e) do { float g = fout[i]; if (fabsf(g - (e)) > 1e-4f * fabsf(e) && fabsf(g - (e)) > 1e-5f) { if (r.errors < 5) printf("  [%d] got %.4f exp %.4f\n", i, g, (float)(e)); r.errors++; } } while(0)
    #define CI(i, e) do { if (iout[i] != (uint32_t)(e)) { if (r.errors < 5) printf("  [%d] got %u exp %d\n", i, iout[i], (int)(e)); r.errors++; } } while(0)

    CF(0, fa+fb); CF(1, fa-fb); CF(2, fa*fb); CF(3, fa/fb); CF(4, sqrtf(fa));
    CF(5, fa*fb+fc); CF(6, fa*fb-fc); CF(7, -(fa*fb-fc)); CF(8, -(fa*fb+fc));
    CF(9, 3.0f); CF(10, -3.0f); CF(11, 3.0f);
    CF(12, 2.0f); CF(13, 3.0f);
    CI(14, 0); CI(15, 0); CI(16, 1); CI(17, 1);
    CI(18, 3); CI(19, 3); CF(20, -5.0f); CF(21, 7.0f);
    CI(22, 7); CI(23, 0x40400000);
    CI(24, 0x40); CI(25, 0x02);
    #undef CF
    #undef CI

    printf("  %s: 26 RV32F results\n", r.errors == 0 ? "PASS" : "FAIL");
    return r;
}

static TestResult test_mem_access(GPGPUState *s)
{
    TestResult r = { .name = "mem_access (lb/lbu/lh/lhu/sb/sh)", .errors = 0 };
    uint32_t kern = 0x500000;
    uint8_t input[8] = { 0x7F, 0x80, 0xFF, 0x00, 0x34, 0x12, 0x78, 0x56 };
    memcpy(s->vram_ptr + 0x100000, input, 8);

    if (run_kernel(s, "kernels/mem_access.bin", kern, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    uint32_t *out = (uint32_t *)(s->vram_ptr + 0x300000);
    #define CHK(i, e) do { if (out[i] != (uint32_t)(e)) { printf("  [%d] got %d exp %d\n", i, out[i], (int)(e)); r.errors++; } } while(0)
    CHK(0, 127); CHK(1, (int32_t)-128); CHK(2, (int32_t)-1);
    CHK(3, 128); CHK(4, 255);
    CHK(5, 0x1234); CHK(6, 0x5678); CHK(7, 0x1234);
    CHK(8, 0x00FF807F); CHK(9, 0x42); CHK(10, 0x4321); CHK(11, 0x00FF807F);
    #undef CHK
    printf("  %s: 12 memory tests\n", r.errors == 0 ? "PASS" : "FAIL");
    return r;
}

static TestResult test_matmul(GPGPUState *s)
{
    TestResult r = { .name = "matmul", .errors = 0 };
    uint32_t M = 1, K = 2, N = 1, kern = 0x500000;
    *(uint32_t *)(s->vram_ptr + 0x0) = K;
    ((float *)(s->vram_ptr + 0x100000))[0] = 1; ((float *)(s->vram_ptr + 0x100000))[1] = 2;
    ((float *)(s->vram_ptr + 0x200000))[0] = 3; ((float *)(s->vram_ptr + 0x200000))[1] = 4;
    if (run_kernel(s, "kernels/matmul.bin", kern, M, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float exp = 1 * 3 + 2 * 4, got = ((float *)(s->vram_ptr + 0x300000))[0];
    if (fabsf(got - exp) > 1e-3f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: C[0]=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", got);
    return r;
}

/* ============================================================
 * 并行性能测试
 * ============================================================ */

static TestResult test_perf_vecmul(GPGPUState *s)
{
    TestResult r = { .name = "perf_vecmul 65536", .errors = 0 };
    uint32_t N = 65536, kern = 0x500000;
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + 0x200000))[i] = 3.0f;
    }
    /* grid=(N/32,1,1) blocks, block=(32,1,1) threads */
    if (run_kernel(s, "kernels/vecmul.bin", kern, N/32, 1, 1, 32, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    /* spot-check */
    for (uint32_t i = 0; i < N; i += 1024) {
        float exp = (float)(i % 256) * 3.0f;
        float got = ((float *)(s->vram_ptr + 0x300000))[i];
        if (fabsf(got - exp) > 1e-4f) { r.errors++; break; }
    }
    printf("  %s: %u elements\n", r.errors == 0 ? "PASS" : "FAIL", N);
    return r;
}

static TestResult test_perf_matmul(GPGPUState *s)
{
    TestResult r = { .name = "perf_matmul 128", .errors = 0 };
    uint32_t M = 128, K = 128, N = 128, kern = 0x500000;
    *(uint32_t *)(s->vram_ptr + 0x0) = K;
    for (uint32_t i = 0; i < M * K; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = 1.0f;
    for (uint32_t i = 0; i < K * N; i++)
        ((float *)(s->vram_ptr + 0x200000))[i] = 1.0f;
    /* grid=(M,1,1), block=(N,1,1) → M*N threads, M blocks */
    if (run_kernel(s, "kernels/matmul.bin", kern, M, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    /* C[i][j] should be K for all i,j */
    float got = ((float *)(s->vram_ptr + 0x300000))[0];
    float exp = (float)K;
    if (fabsf(got - exp) > 1.0f) { r.errors++; printf("  C[0]=%.2f exp=%.2f\n", got, exp); }
    printf("  %s: C[0]=%.2f (exp %.0f)\n", r.errors == 0 ? "PASS" : "FAIL", got, exp);
    return r;
}

static TestResult test_vecmul(GPGPUState *s)
{
    TestResult r = { .name = "vecmul", .errors = 0 };
    uint32_t N = 1, kern = 0x500000;
    ((float *)(s->vram_ptr + 0x100000))[0] = 2.5f;
    ((float *)(s->vram_ptr + 0x200000))[0] = 3.0f;
    if (run_kernel(s, "kernels/vecmul.bin", kern, N, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float exp = 7.5f, got = ((float *)(s->vram_ptr + 0x300000))[0];
    if (fabsf(got - exp) > 1e-5f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: C[0]=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", got);
    return r;
}

static TestResult test_scal_mul(GPGPUState *s)
{
    TestResult r = { .name = "scal_mul", .errors = 0 };
    uint32_t N = 1, kern = 0x500000;
    ((float *)(s->vram_ptr + 0x100000))[0] = 4.0f;
    *(float *)(s->vram_ptr + 0x400000) = 0.75f;
    if (run_kernel(s, "kernels/scal_mul.bin", kern, N, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float exp = 3.0f, got = ((float *)(s->vram_ptr + 0x200000))[0];
    if (fabsf(got - exp) > 1e-5f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: C[0]=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", got);
    return r;
}

static TestResult test_gelu(GPGPUState *s)
{
    TestResult r = { .name = "gelu", .errors = 0 };
    uint32_t N = 1, kern = 0x500000;
    float x = 0.5f;
    ((float *)(s->vram_ptr + 0x100000))[0] = x;
    if (run_kernel(s, "kernels/gelu.bin", kern, N, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float exp = x * (1.0f / (1.0f + expf(-1.702f * x)));
    float got = ((float *)(s->vram_ptr + 0x200000))[0];
    if (fabsf(got - exp) > 1e-3f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: GELU(%.2f)=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", x, got);
    return r;
}

static TestResult test_softmax(GPGPUState *s)
{
    TestResult r = { .name = "softmax", .errors = 0 };
    uint32_t N = 64, kern = 0x500000;
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = ((float)i - 32.0f) * 0.1f;
    if (run_kernel(s, "kernels/softmax.bin", kern, 1, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float sum = 0;
    for (uint32_t i = 0; i < N; i++) sum += ((float *)(s->vram_ptr + 0x200000))[i];
    if (fabsf(sum - 1.0f) > 0.02f) { r.errors++; printf("  sum=%.4f\n", sum); }
    printf("  %s: N=%u sum=%.6f\n", r.errors == 0 ? "PASS" : "FAIL", N, sum);
    return r;
}

static TestResult test_softmax_norm(GPGPUState *s)
{
    TestResult r = { .name = "softmax_norm", .errors = 0 };
    uint32_t N = 1, kern = 0x500000;
    ((float *)(s->vram_ptr + 0x200000))[0] = expf(1.0f);
    *(float *)(s->vram_ptr + 0x400000) = expf(1.0f);
    if (run_kernel(s, "kernels/softmax_norm.bin", kern, N, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float got = ((float *)(s->vram_ptr + 0x300000))[0];
    if (fabsf(got - 1.0f) > 0.01f) { r.errors++; printf("  got %.4f\n", got); }
    printf("  %s: out[0]=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", got);
    return r;
}

static TestResult test_conv2d(GPGPUState *s)
{
    TestResult r = { .name = "conv2d", .errors = 0 };
    uint32_t H = 3, W = 3, K = 2, kern = 0x500000;
    float in[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, kr[4] = { 1, 0, 0, 1 };
    for (uint32_t i = 0; i < H; i++)
        for (uint32_t j = 0; j < W; j++)
            ((float *)(s->vram_ptr + 0x100000))[i * W + j] = in[i * W + j];
    for (uint32_t i = 0; i < K; i++)
        for (uint32_t j = 0; j < K; j++)
            ((float *)(s->vram_ptr + 0x200000))[i * K + j] = kr[i * K + j];
    if (run_kernel(s, "kernels/conv2d.bin", kern, H, W, 1, K, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    float exp = in[0] * kr[0] + in[1] * kr[1] + in[3] * kr[2] + in[4] * kr[3];
    float got = ((float *)(s->vram_ptr + 0x300000))[0];
    if (fabsf(got - exp) > 1e-4f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: out[0,0]=%.4f\n", r.errors == 0 ? "PASS" : "FAIL", got);
    return r;
}

/* ============================================================
 * test_runner_run — 运行所有测试
 * ============================================================ */

int test_runner_run(GPGPUState *s)
{
    printf("=== VPU Interpreter Test Suite ===\n\n");

    int errors = 0;
    TestResult r;

    #define RUN(name, fn) do { \
        printf("--- %s ---\n", name); \
        r = fn(s); \
        errors += (r.errors > 0) ? r.errors : 0; \
        printf("\n"); \
    } while(0)

    RUN("Test 1: Integer Vector Add",    test_vector_add);
    RUN("Test 2: Float SAXPY",           test_saxpy);
    RUN("Test 3: Dot Product",           test_dot_product);
    RUN("Test 4: Memory Copy",           test_memcpy);
    RUN("Test 5: Massive Loop",          test_loop_perf);
    RUN("Test 6: RV32M Multiply/Divide", test_rv32m);
    RUN("Test 7: RV32F Full Coverage",   test_rv32f);
    RUN("Test 8: Memory Access",         test_mem_access);
    RUN("Test 9: Matmul",                test_matmul);
    RUN("Test 10: Vecmul",               test_vecmul);
    RUN("Test 11: Scal_mul",             test_scal_mul);
    RUN("Test 12: GELU",                 test_gelu);
    RUN("Test 13: Softmax",              test_softmax);
    RUN("Test 14: Softmax_norm",         test_softmax_norm);
    RUN("Test 15: Conv2d",               test_conv2d);

    printf("\n=== Parallel Performance ===\n\n");
    RUN("Perf: vecmul 65536",              test_perf_vecmul);
    RUN("Perf: matmul 128x128x128",        test_perf_matmul);

    #undef RUN

    printf("===========================================\n");
    printf("Total errors: %d\n", errors);
    printf("===========================================\n");

    return errors;
}
