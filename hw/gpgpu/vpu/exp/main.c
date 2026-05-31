/*
 * main.c — RISC-V SIMT 解释器独立性能测试
 *
 * 加载测试 kernel，初始化数据，运行并测量性能。
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

/* ============================================================
 * 辅助函数
 * ============================================================ */

/* 将文件读入堆内存，返回 malloc 的 buffer 和大小。
 * caller 负责 free(buf)。 */
static uint8_t *read_file(const char *path, size_t *out_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return NULL;
    }
    *out_size = st.st_size;
    uint8_t *buf = malloc(st.st_size);
    if (!buf) {
        perror("malloc");
        close(fd);
        return NULL;
    }
    ssize_t n = read(fd, buf, st.st_size);
    close(fd);
    if (n < 0 || (size_t)n != st.st_size) {
        perror("read");
        free(buf);
        return NULL;
    }
    return buf;
}

/* 将 kernel .bin 文件载入 VRAM 指定偏移处。
 * kernel_addr 指出 VRAM 中的起始地址。 */
static size_t load_kernel(const char *path, GPGPUState *s, uint32_t kernel_addr)
{
    size_t size;
    uint8_t *buf = read_file(path, &size);
    if (!buf) return 0;

    if (kernel_addr + size > s->vram_size) {
        fprintf(stderr, "ERROR: kernel too large for VRAM "
                        "(addr=0x%x size=%zu vram=%lu)\n",
                kernel_addr, size, (unsigned long)s->vram_size);
        free(buf);
        return 0;
    }
    memcpy(s->vram_ptr + kernel_addr, buf, size);
    printf("  Loaded %s: %zu bytes at VRAM[0x%x]\n", path, size, kernel_addr);
    free(buf);
    return size;
}

/* ============================================================
 * 测试框架
 * =========================================================== */

typedef struct {
    const char *name;
    int   errors;
    double elapsed;
} TestResult;

/* 运行 kernel 5 次，对比旧/新解释器，统计平均 */
#define N_RUNS 5
#define SAVE_SIZE (4*1024*1024 + 4096)

static int run_kernel(GPGPUState *s, const char *kern_path,
                       uint32_t kernel_addr,
                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                       uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
    struct timespec t0, t1;
    double old_sum = 0, fast_sum = 0, th_sum = 0, simd_sum = 0;
    int ret = 0, total_mismatch = 0;

    s->kernel.kernel_addr = kernel_addr;
    s->kernel.grid_dim[0] = grid_x;
    s->kernel.grid_dim[1] = grid_y;
    s->kernel.grid_dim[2] = grid_z;
    s->kernel.block_dim[0] = block_x;
    s->kernel.block_dim[1] = block_y;
    s->kernel.block_dim[2] = block_z;

    uint8_t *saved_in = malloc(SAVE_SIZE);
    uint8_t *fast_out = malloc(SAVE_SIZE);
    uint8_t *th_out   = malloc(SAVE_SIZE);
    uint8_t *simd_out = malloc(SAVE_SIZE);
    uint8_t *old_out  = malloc(SAVE_SIZE);

    for (int run = 0; run < N_RUNS; run++) {
        memcpy(saved_in, s->vram_ptr, SAVE_SIZE);

        /* threaded */
        load_kernel(kern_path, s, kernel_addr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int r3 = gpgpu_core_threaded_exec_kernel(s);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        th_sum += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        memcpy(th_out, s->vram_ptr, SAVE_SIZE);

        /* simd */
        memcpy(s->vram_ptr, saved_in, SAVE_SIZE);
        load_kernel(kern_path, s, kernel_addr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int r4 = gpgpu_core_simd_exec_kernel(s);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        simd_sum += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        memcpy(simd_out, s->vram_ptr, SAVE_SIZE);

        /* fast */
        memcpy(s->vram_ptr, saved_in, SAVE_SIZE);
        load_kernel(kern_path, s, kernel_addr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int r2 = gpgpu_core_fast_exec_kernel(s);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fast_sum += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        memcpy(fast_out, s->vram_ptr, SAVE_SIZE);

        /* old */
        memcpy(s->vram_ptr, saved_in, SAVE_SIZE);
        load_kernel(kern_path, s, kernel_addr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ret = gpgpu_core_exec_kernel(s);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        old_sum += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        memcpy(old_out, s->vram_ptr, SAVE_SIZE);

        if (r2 != 0 || r3 != 0 || r4 != 0) ret = -1;
        int mm = 0;
        for (size_t i = 0; i < SAVE_SIZE; i += 4) {
            uint32_t tv=*(uint32_t*)(th_out+i),fv=*(uint32_t*)(fast_out+i),ov=*(uint32_t*)(old_out+i),sv=*(uint32_t*)(simd_out+i);
            if(tv!=ov||fv!=ov||sv!=ov) {
                if (mm < 5) printf("  run#%d DIFF @ 0x%06zx: th=0x%08x fast=0x%08x old=0x%08x simd=0x%08x\n", run, i, tv, fv, ov, sv);
                mm++;
            }
        }
        total_mismatch += mm; if (mm > 0) ret = -1;
        memcpy(s->vram_ptr, saved_in, SAVE_SIZE);
    }
    memcpy(s->vram_ptr, saved_in, SAVE_SIZE);
    load_kernel(kern_path, s, kernel_addr);
    gpgpu_core_exec_kernel(s);

    double oa=old_sum/N_RUNS,fa=fast_sum/N_RUNS,ta=th_sum/N_RUNS,sa=simd_sum/N_RUNS;
    printf("  Old %5.0fus | Fast %5.0fus(+%.0f%%) | Th %5.0fus(+%.0f%%) | SIMD %5.0fus(+%.0f%%) | %s\n",
           oa*1e6,fa*1e6,(oa/fa-1)*100,ta*1e6,(oa/ta-1)*100,sa*1e6,(oa/sa-1)*100,
           total_mismatch==0?"MATCH":"DIFF!");

    return ret;
}

/* ============================================================
 * 测试用例
 * ============================================================ */

/* 测试 1: 多 lane 整数向量加法 (vecmul kernel, block_dim=N) */
static TestResult test_vector_add(GPGPUState *s)
{
    TestResult r = { .name = "vecmul (multi-lane)", .errors = 0 };
    uint32_t N = 2048; /* 2048 元素，32 lane × 64 warp */
    uint32_t kern_addr = 0x500000;

    for (uint32_t i = 0; i < N; i++) {
        ((float*)(s->vram_ptr + 0x100000))[i] = (float)(i + 1);
        ((float*)(s->vram_ptr + 0x200000))[i] = 2.0f;
    }

    if (run_kernel(s, "kernels/vecmul.bin", kern_addr, 1, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }

    for (uint32_t i = 0; i < N; i++) {
        float expected = (float)(i + 1) * 2.0f;
        float got = ((float*)(s->vram_ptr + 0x300000))[i];
        if (fabsf(got - expected) > 1e-5f) {
            if (r.errors < 5) printf("  [%u] exp %.1f got %.1f\n", i, expected, got);
            r.errors++;
        }
    }
    if (r.errors == 0) printf("  PASS: %u elements (multi-lane)\n", N);
    else printf("  FAIL: %d errors\n", r.errors);
    return r;
}

/* 测试 2: 单精度浮点 SAXPY */
static TestResult test_saxpy(GPGPUState *s)
{
    TestResult r = { .name = "saxpy (float fmadd)", .errors = 0 };

    uint32_t N = 16384; /* 16K 元素 */
    float A = 2.5f;
    uint32_t kern_addr = 0x500000;

    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    *(float *)(s->vram_ptr + 0x4) = A;

    /* 填充随机浮点数 */
    srand48(42);
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(drand48() * 100.0 - 50.0);  /* X */
        ((float *)(s->vram_ptr + 0x200000))[i] = (float)(drand48() * 100.0 - 50.0);  /* Y */
    }

    /* 保存 Y 的副本用于验证 */
    float *y_copy = malloc(N * sizeof(float));
    memcpy(y_copy, s->vram_ptr + 0x200000, N * sizeof(float));

    if (run_kernel(s, "kernels/saxpy.bin", kern_addr, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1;
        free(y_copy);
        return r;
    }

    /* 验证: Y[i] = A * X[i] + Y_old[i] */
    for (uint32_t i = 0; i < N; i++) {
        float x = ((float *)(s->vram_ptr + 0x100000))[i];
        float expected = A * x + y_copy[i];
        float got = ((float *)(s->vram_ptr + 0x200000))[i];
        /* 允许微小的浮点误差 */
        if (fabsf(got - expected) > 1e-5f * fabsf(expected) && fabsf(got - expected) > 1e-6f) {
            if (r.errors < 10)
                printf("  Mismatch at [%u]: expected %.6f, got %.6f\n",
                       i, expected, got);
            r.errors++;
        }
    }

    free(y_copy);

    if (r.errors == 0)
        printf("  PASS: %u elements verified\n", N);
    else
        printf("  FAIL: %d errors / %u elements\n", r.errors, N);

    uint32_t est_insts = 10 + N * 13; /* FP 指令更多 */
    printf("  ~%u instructions (estimated)\n", est_insts);

    return r;
}

/* 测试 4: 多 lane 大规模吞吐量 */
static TestResult test_loop_perf(GPGPUState *s)
{
    TestResult r = { .name = "massive (multi-lane)", .errors = 0 };
    uint32_t N = 2048;
    uint32_t kern_addr = 0x500000;

    for (uint32_t i = 0; i < N; i++) {
        ((float*)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
        ((float*)(s->vram_ptr + 0x200000))[i] = 3.0f;
    }

    if (run_kernel(s, "kernels/vecmul.bin", kern_addr, 1, 1, 1, N, 1, 1) != 0) {
        r.errors = -1; return r;
    }
    printf("  ~%u elements (multi-lane)\n", N);
    return r;
}

/* 测试 5: RV32M 乘除扩展 */
static TestResult test_rv32m(GPGPUState *s)
{
    TestResult r = { .name = "rv32m (mul/div/rem)", .errors = 0 };

    uint32_t kern_addr = 0x500000;

    /* 测试用例: {a, b} 对
     * case 0: 正×正   case 1: 正×负   case 2: 负×负
     * case 3: 除零    case 4: 正常除法 case 5: 边界值 */
    uint32_t test_cases[][2] = {
        {10, 3},           /* 0: mul=30, div=3, rem=1 */
        {10, (uint32_t)-3}, /* 1: mul=-30, div=-3, rem=1 */
        {(uint32_t)-5, (uint32_t)-2}, /* 2: mul=10, div=2, rem=-1 */
        {100, 0},           /* 3: div/rem by zero */
        {0x80000000, 2},    /* 4: INT_MIN / 2 */
        {0xFFFFFFFF, 0xFFFFFFFF}, /* 5: all-ones * all-ones */
    };
    uint32_t N = sizeof(test_cases) / sizeof(test_cases[0]);

    *(uint32_t *)(s->vram_ptr + 0x0) = N;
    memcpy(s->vram_ptr + 0x100000, test_cases, sizeof(test_cases));

    if (run_kernel(s, "kernels/rv32m.bin", kern_addr, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }

    /* 验证 */
    for (uint32_t i = 0; i < N; i++) {
        int32_t  a = test_cases[i][0];
        int32_t  b = test_cases[i][1];
        uint32_t base = i * 8;
        uint32_t *out = (uint32_t *)(s->vram_ptr + 0x300000);

        uint32_t exp_mul    = (uint32_t)(a * b);
        uint32_t exp_mulh   = (uint32_t)(((int64_t)a * (int64_t)b) >> 32);
        uint32_t exp_mulhsu = (uint32_t)(((int64_t)a * (uint64_t)(uint32_t)b) >> 32);
        uint32_t exp_mulhu  = (uint32_t)(((uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b) >> 32);
        uint32_t exp_div    = (b == 0) ? (uint32_t)-1 : (uint32_t)(a / b);
        uint32_t exp_divu   = (b == 0) ? 0xFFFFFFFF : ((uint32_t)a / (uint32_t)b);
        uint32_t exp_rem    = (b == 0) ? (uint32_t)a : (uint32_t)(a % b);
        uint32_t exp_remu   = (b == 0) ? (uint32_t)a : ((uint32_t)a % (uint32_t)b);

        #define CHK(idx, name, exp) do { \
            if (out[base+(idx)] != (exp)) { \
                if (r.errors < 5) printf("  [%u] %s: got 0x%x exp 0x%x\n", i, name, out[base+(idx)], (exp)); \
                r.errors++; }} while(0)

        CHK(0, "MUL   ", exp_mul);
        CHK(1, "MULH  ", exp_mulh);
        CHK(2, "MULHSU", exp_mulhsu);
        CHK(3, "MULHU ", exp_mulhu);
        CHK(4, "DIV   ", exp_div);
        CHK(5, "DIVU  ", exp_divu);
        CHK(6, "REM   ", exp_rem);
        CHK(7, "REMU  ", exp_remu);
        #undef CHK
    }

    if (r.errors == 0) printf("  PASS: all 8×%u instructions verified\n", N);
    else printf("  FAIL: %d errors\n", r.errors);
    return r;
}

/* 测试 6: RV32F 浮点全覆盖 */
static TestResult test_rv32f(GPGPUState *s)
{
    TestResult r = { .name = "rv32f (fma/cmp/cvt/class)", .errors = 0 };
    uint32_t kern_addr = 0x500000;

    /* 输入: a=3.0f, b=2.0f, c=4.0f, i32=-5, u32=7 */
    float    fa = 3.0f, fb = 2.0f, fc = 4.0f;
    int32_t  i32 = -5;
    uint32_t u32 = 7;

    memcpy(s->vram_ptr + 0x100000, &fa, 4);
    memcpy(s->vram_ptr + 0x100004, &fb, 4);
    memcpy(s->vram_ptr + 0x100008, &fc, 4);
    memcpy(s->vram_ptr + 0x10000C, &i32, 4);
    memcpy(s->vram_ptr + 0x100010, &u32, 4);

    if (run_kernel(s, "kernels/rv32f.bin", kern_addr, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }

    float    *fout = (float *)(s->vram_ptr + 0x300000);
    uint32_t *iout = (uint32_t *)(s->vram_ptr + 0x300000);

    /* 按 kernel 输出顺序验证 */
    #define CHECK_F(idx, exp, label) do { \
        float got = fout[idx]; float e = (exp); \
        if (fabsf(got - e) > 1e-4f * fabsf(e) && fabsf(got - e) > 1e-5f) { \
            if (r.errors < 8) printf("  %s: got %.4f exp %.4f\n", label, got, e); \
            r.errors++; }} while(0)

    CHECK_F(0,  fa+fb,       "FADD   3+2");
    CHECK_F(1,  fa-fb,       "FSUB   3-2");
    CHECK_F(2,  fa*fb,       "FMUL   3*2");
    CHECK_F(3,  fa/fb,       "FDIV   3/2");
    CHECK_F(4,  sqrtf(fa),   "FSQRT  sqrt(3)");
    /* FMA */
    CHECK_F(5,  fa*fb+fc,    "FMADD  3*2+4");
    CHECK_F(6,  fa*fb-fc,    "FMSUB  3*2-4");
    CHECK_F(7,  -(fa*fb-fc), "FNMSUB -(3*2-4)");
    CHECK_F(8,  -(fa*fb+fc), "FNMADD -(3*2+4)");
    /* 符号注入 */
    CHECK_F(9,  3.0f,        "FSGNJ  +|3|");
    CHECK_F(10, -3.0f,       "FSGNJN -|3|");
    CHECK_F(11, 3.0f,        "FSGNJX 3^+");
    /* 最值 */
    CHECK_F(12, 2.0f,        "FMIN   min(3,2)");
    CHECK_F(13, 3.0f,        "FMAX   max(3,2)");

    /* 比较 (整数输出) */
    #define CHECK_I(arr_idx, exp_val, label) do { \
        if (iout[arr_idx] != (uint32_t)(exp_val)) { \
            if (r.errors < 8) printf("  %s: got %u exp %d\n", label, iout[arr_idx], (int)(exp_val)); \
            r.errors++; }} while(0)

    CHECK_I(14, 0, "FEQ   3==2");
    CHECK_I(15, 0, "FLT   3<2");
    CHECK_I(16, 1, "FLE   2<=3");
    CHECK_I(17, 1, "FLT   2<3");

    /* 转换 */
    CHECK_I(18, 3,             "FCVT.W.S   (int)3");
    CHECK_I(19, 3,             "FCVT.WU.S  (uint)3");
    CHECK_F(20, -5.0f,         "FCVT.S.W   (flt)-5");
    CHECK_F(21, 7.0f,          "FCVT.S.WU  (flt)7");

    /* 数据移动 */
    CHECK_I(22, 7,             "FMV.W.X  bits(7)");  /* float bits of 7 = 7 */
    CHECK_I(23, 0x40400000,    "FMV.X.W  bits(3.0f)"); /* 3.0f bits = 0x40400000 */

    /* 分类: fclass(3.0f) → positive normal = bit6 = 0x40 = 64 */
    CHECK_I(24, 0x40,          "FCLASS   3.0f");
    CHECK_I(25, 0x02,          "FCLASS   -3.0f");
    #undef CHECK_F
    #undef CHECK_I

    if (r.errors == 0) printf("  PASS: all 26 RV32F results verified\n");
    else printf("  FAIL: %d errors\n", r.errors);
    return r;
}

/* 测试 7: 字节/半字访存 */
static TestResult test_mem_access(GPGPUState *s)
{
    TestResult r = { .name = "mem_access (lb/lbu/lh/lhu/sb/sh)", .errors = 0 };
    uint32_t kern_addr = 0x500000;

    /* 输入: 构造特定字节序列 (小端序)
     *   字节: [0x7F, 0x80, 0xFF, 0x00, 0x34, 0x12, 0x78, 0x56] */
    uint8_t input[8] = {0x7F, 0x80, 0xFF, 0x00, 0x34, 0x12, 0x78, 0x56};
    memcpy(s->vram_ptr + 0x100000, input, 8);

    if (run_kernel(s, "kernels/mem_access.bin", kern_addr, 1, 1, 1, 1, 1, 1) != 0) {
        r.errors = -1; return r;
    }

    uint32_t *out = (uint32_t *)(s->vram_ptr + 0x300000);

    #define CHK(idx, exp, label) do { \
        if (out[idx] != (uint32_t)(exp)) { \
            printf("  %s: got %d (0x%x) exp %d\n", label, out[idx], out[idx], (int)(exp)); \
            r.errors++; }} while(0)

    CHK(0,  127,       "LB   0x7F");
    CHK(1,  (int32_t)-128, "LB   0x80");
    CHK(2,  (int32_t)-1,   "LB   0xFF");
    CHK(3,  128,       "LBU  0x80");
    CHK(4,  255,       "LBU  0xFF");
    CHK(5,  0x1234,    "LH   0x1234");
    CHK(6,  0x5678,    "LH   0x5678");
    CHK(7,  0x1234,    "LHU  0x1234");
    CHK(8,  0x00FF807F,"LW   word0");   /* 小端: {0x7F,0x80,0xFF,0x00} → 0x00FF807F */
    CHK(9,  0x42,      "SB   readback");
    CHK(10, 0x4321,    "SH   readback");
    CHK(11, 0x00FF807F,"SW   readback");
    #undef CHK

    /* Fix LW expected: offset 0 bytes = {0x7F, 0x80, 0xFF, 0x00}
     * 小端: 0x00FF807F */
    /* Already corrected above */

    if (r.errors == 0) printf("  PASS: all 12 memory access tests verified\n");
    else printf("  FAIL: %d errors\n", r.errors);
    return r;
}



/* 测试 5: 矩阵乘法 */
static TestResult test_matmul(GPGPUState *s)
{
    TestResult r = {.name="matmul", .errors=0};
    uint32_t M=1,K=2,N=1,kern=0x500000;
    ((float*)(s->vram_ptr+0x100000))[0]=1; ((float*)(s->vram_ptr+0x100000))[1]=2;
    ((float*)(s->vram_ptr+0x200000))[0]=3; ((float*)(s->vram_ptr+0x200000))[1]=4;
    if(run_kernel(s,"kernels/matmul.bin",kern,M,K,1,N,1,1)!=0){r.errors=-1;return r;}
    float exp=1*3+2*4,got=((float*)(s->vram_ptr+0x300000))[0];
    if(fabsf(got-exp)>1e-3f){r.errors++;printf("  exp %.4f got %.4f\n",exp,got);}
    printf("  %s: C[0]=%.4f\n",r.errors==0?"PASS":"FAIL",got);
    return r;
}

/* 测试 3: 浮点向量内积 */
static TestResult test_dot_product(GPGPUState *s)
{
    TestResult r = {.name="dot_product", .errors=0};
    uint32_t N = 32768, kern = 0x500000;
    *(uint32_t*)(s->vram_ptr + 0x0) = N;
    float sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        float a = (float)(i % 100) * 0.01f;
        float b = (float)((i+1) % 100) * 0.01f;
        ((float*)(s->vram_ptr + 0x100000))[i] = a;
        ((float*)(s->vram_ptr + 0x200000))[i] = b;
        sum += a * b;
    }
    if (run_kernel(s, "kernels/dot_product.bin", kern, 1,1,1, 1,1,1) != 0) { r.errors=-1; return r; }
    float got = *(float*)(s->vram_ptr + 0x4);
    if (fabsf(got - sum) > 0.01f) { r.errors++; printf("  exp %.4f got %.4f\n", sum, got); }
    printf("  %s: N=%u result=%.4f\n", r.errors==0?"PASS":"FAIL", N, got);
    return r;
}

/* 测试 4: 内存拷贝带宽 */
static TestResult test_memcpy(GPGPUState *s)
{
    TestResult r = {.name="memcpy", .errors=0};
    uint32_t N = 131072, kern = 0x500000;
    *(uint32_t*)(s->vram_ptr + 0x0) = N;
    for (uint32_t i = 0; i < N; i++) {
        ((uint32_t*)(s->vram_ptr + 0x100000))[i] = i;
        ((uint32_t*)(s->vram_ptr + 0x200000))[i] = 0;
    }
    if (run_kernel(s, "kernels/memcpy.bin", kern, 1,1,1, 1,1,1) != 0) { r.errors=-1; return r; }
    for (uint32_t i = 0; i < N; i++) {
        if (((uint32_t*)(s->vram_ptr + 0x200000))[i] != i) { r.errors++; break; }
    }
    uint64_t bytes = (uint64_t)N * 8; /* 4B read + 4B write */
    printf("  %s: %lu bytes copied\n", r.errors==0?"PASS":"FAIL", (unsigned long)bytes);
    return r;
}

/* 测试 5: 矩阵乘法 (guest kernel) */

/* 测试 8-14: guest kernel 测试 */


/* 测试 8: vecmul */
static TestResult test_vecmul(GPGPUState *s)
{
    TestResult r = {.name="vecmul", .errors=0};
    uint32_t N = 1, kern = 0x500000;
    ((float*)(s->vram_ptr+0x100000))[0] = 2.5f;
    ((float*)(s->vram_ptr+0x200000))[0] = 3.0f;
    if (run_kernel(s, "kernels/vecmul.bin", kern, N, 1, 1, 1, 1, 1) != 0) { r.errors=-1; return r; }
    float exp = 7.5f, got = ((float*)(s->vram_ptr+0x300000))[0];
    if (fabsf(got-exp) > 1e-5f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: C[0]=%.4f\n", r.errors==0?"PASS":"FAIL", got);
    return r;
}

/* 测试 9: scal_mul */
static TestResult test_scal_mul(GPGPUState *s)
{
    TestResult r = {.name="scal_mul", .errors=0};
    uint32_t N = 1, kern = 0x500000;
    ((float*)(s->vram_ptr+0x100000))[0] = 4.0f;
    *(float*)(s->vram_ptr+0x400000) = 0.75f;
    if (run_kernel(s, "kernels/scal_mul.bin", kern, N, 1, 1, 1, 1, 1) != 0) { r.errors=-1; return r; }
    float exp = 3.0f, got = ((float*)(s->vram_ptr+0x200000))[0];
    if (fabsf(got-exp) > 1e-5f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: C[0]=%.4f\n", r.errors==0?"PASS":"FAIL", got);
    return r;
}

/* 测试 10: gelu */
static TestResult test_gelu(GPGPUState *s)
{
    TestResult r = {.name="gelu", .errors=0};
    uint32_t N = 1, kern = 0x500000;
    float x = 0.5f;
    ((float*)(s->vram_ptr+0x100000))[0] = x;
    if (run_kernel(s, "kernels/gelu.bin", kern, N, 1, 1, 1, 1, 1) != 0) { r.errors=-1; return r; }
    float exp = x * (1.0f / (1.0f + expf(-1.702f * x)));
    float got = ((float*)(s->vram_ptr+0x200000))[0];
    if (fabsf(got-exp) > 1e-3f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: GELU(%.2f)=%.4f\n", r.errors==0?"PASS":"FAIL", x, got);
    return r;
}

/* 测试 11: softmax */
static TestResult test_softmax(GPGPUState *s)
{
    TestResult r = {.name="softmax", .errors=0};
    uint32_t N = 64, kern = 0x500000;
    for (uint32_t i = 0; i < N; i++)
        ((float*)(s->vram_ptr+0x100000))[i] = ((float)i - 32.0f) * 0.1f;
    if (run_kernel(s, "kernels/softmax.bin", kern, 1,1,1, N,1,1) != 0) { r.errors=-1; return r; }
    float sum = 0;
    for (uint32_t i = 0; i < N; i++) sum += ((float*)(s->vram_ptr+0x200000))[i];
    if (fabsf(sum - 1.0f) > 0.02f) { r.errors++; printf("  sum=%.4f expect ~1\n", sum); }
    printf("  %s: N=%u sum=%.6f\n", r.errors==0?"PASS":"FAIL", N, sum);
    return r;
}

/* 测试 12: softmax_norm */
static TestResult test_softmax_norm(GPGPUState *s)
{
    TestResult r = {.name="softmax_norm", .errors=0};
    uint32_t N = 1, kern = 0x500000;
    ((float*)(s->vram_ptr+0x200000))[0] = expf(1.0f);
    *(float*)(s->vram_ptr+0x400000) = expf(1.0f);
    if (run_kernel(s, "kernels/softmax_norm.bin", kern, N, 1, 1, 1, 1, 1) != 0) { r.errors=-1; return r; }
    float got = ((float*)(s->vram_ptr+0x300000))[0];
    if (fabsf(got - 1.0f) > 0.01f) { r.errors++; printf("  got %.4f expect 1\n", got); }
    printf("  %s: out[0]=%.4f\n", r.errors==0?"PASS":"FAIL", got);
    return r;
}

/* 测试 14: conv2d */
static TestResult test_conv2d(GPGPUState *s)
{
    TestResult r = {.name="conv2d", .errors=0};
    uint32_t H = 3, W = 3, K = 2, kern = 0x500000;
    float in[9] = {1,2,3,4,5,6,7,8,9}, kr[4] = {1,0,0,1};
    for (uint32_t i = 0; i < H; i++)
        for (uint32_t j = 0; j < W; j++)
            ((float*)(s->vram_ptr+0x100000))[i*W+j] = in[i*W+j];
    for (uint32_t i = 0; i < K; i++)
        for (uint32_t j = 0; j < K; j++)
            ((float*)(s->vram_ptr+0x200000))[i*K+j] = kr[i*K+j];
    if (run_kernel(s, "kernels/conv2d.bin", kern, H, W, 1, K, 1, 1) != 0) { r.errors=-1; return r; }
    float exp = in[0]*kr[0] + in[1]*kr[1] + in[3]*kr[2] + in[4]*kr[3];
    float got = ((float*)(s->vram_ptr+0x300000))[0];
    if (fabsf(got-exp) > 1e-4f) { r.errors++; printf("  exp %.4f got %.4f\n", exp, got); }
    printf("  %s: out[0,0]=%.4f exp=%.4f\n", r.errors==0?"PASS":"FAIL", got, exp);
    return r;

}

/* ============================================================
 * main
 * ============================================================ */

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("=== RISC-V SIMT Interpreter Performance Test ===\n\n");

    /* 初始化设备状态 */
    GPGPUState s;
    memset(&s, 0, sizeof(s));
    s.num_cus = 1;
    s.warps_per_cu = 1;
    s.warp_size = 32;
    s.vram_size = 64 * 1024 * 1024; /* 64 MB */
    s.global_status = 1; /* READY */

    s.vram_ptr = calloc(1, s.vram_size);
    if (!s.vram_ptr) {
        fprintf(stderr, "FATAL: failed to allocate VRAM (%lu MB)\n",
                (unsigned long)(s.vram_size >> 20));
        return 1;
    }
    printf("VRAM: %lu MB allocated at %p\n",
           (unsigned long)(s.vram_size >> 20), (void *)s.vram_ptr);
    printf("Config: %u CU x %u warps/CU x %u lanes/warp\n\n",
           s.num_cus, s.warps_per_cu, s.warp_size);

    /* 运行测试 */
    int total_errors = 0;

    printf("--- Test 1: Integer Vector Add ---\n");
    TestResult r1 = test_vector_add(&s);
    total_errors += (r1.errors > 0) ? r1.errors : 0;
    printf("\n");

    printf("--- Test 2: Float SAXPY ---\n");
    TestResult r2 = test_saxpy(&s);
    total_errors += (r2.errors > 0) ? r2.errors : 0;
    printf("\n");

    printf("--- Test 3: Dot Product (float) ---\n");
    TestResult r_dot = test_dot_product(&s);
    total_errors += (r_dot.errors > 0) ? r_dot.errors : 0;
    printf("\n");

    printf("--- Test 4: Memory Copy (bandwidth) ---\n");
    TestResult r_mem = test_memcpy(&s);
    total_errors += (r_mem.errors > 0) ? r_mem.errors : 0;
    printf("\n");

    printf("--- Test 6: Massive Loop (256K elements) ---\n");
    TestResult r4 = test_loop_perf(&s);
    printf("\n");

    printf("--- Test 5: RV32M Multiply/Divide ---\n");
    TestResult r5 = test_rv32m(&s);
    total_errors += (r5.errors > 0) ? r5.errors : 0;
    printf("\n");

    printf("--- Test 7: RV32F Full Coverage ---\n");
    TestResult r6 = test_rv32f(&s);
    total_errors += (r6.errors > 0) ? r6.errors : 0;
    printf("\n");

    printf("--- Test 7: Memory Access (lb/lbu/lh/lhu/sb/sh) ---\n");
    TestResult r7 = test_mem_access(&s);
    total_errors += (r7.errors > 0) ? r7.errors : 0;
    printf("\n");

    printf("--- Test 5: matmul (guest) ---\n");
    TestResult r_matmul = test_matmul(&s);
    total_errors += (r_matmul.errors > 0) ? r_matmul.errors : 0;
    printf("\n");

    printf("--- Test 8: vecmul (guest) ---\n");
    TestResult r8 = test_vecmul(&s);
    total_errors += (r8.errors > 0) ? r8.errors : 0;
    printf("\n");

    printf("--- Test 9: scal_mul (guest) ---\n");
    TestResult r9 = test_scal_mul(&s);
    total_errors += (r9.errors > 0) ? r9.errors : 0;
    printf("\n");

    printf("--- Test 10: gelu (guest) ---\n");
    TestResult r10 = test_gelu(&s);
    total_errors += (r10.errors > 0) ? r10.errors : 0;
    printf("\n");

    printf("--- Test 13: softmax (guest) ---\n");
    TestResult r11 = test_softmax(&s);
    total_errors += (r11.errors > 0) ? r11.errors : 0;
    printf("\n");

    printf("--- Test 12: softmax_norm (guest) ---\n");
    TestResult r12 = test_softmax_norm(&s);
    total_errors += (r12.errors > 0) ? r12.errors : 0;
    printf("\n");

    printf("--- Test 14: conv2d (guest) ---\n");
    TestResult r14 = test_conv2d(&s);
    total_errors += (r14.errors > 0) ? r14.errors : 0;
    printf("\n");

    /* 总结 */
    printf("===========================================\n");
    printf("RESULTS SUMMARY\n");
    printf("  ---- Performance Tests ----\n");
    printf("  %-45s %s\n", "Test 1: Integer Vector Add",         (r1.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 2: Float SAXPY",               (r2.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 3: Dot Product (float)",       (r_dot.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 4: Memory Copy (bandwidth)",   (r_mem.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 5: Matrix Multiply",          (r_matmul.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 5: Massive Loop (256K)",      (r4.errors == 0) ? "PASS" : "FAIL");
    printf("  ---- Functional Regression ----\n");
    printf("  %-45s %s\n", "Test 4: Massive Loop (256K)",
           (r4.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 6: RV32M (mul/div/rem)",       (r5.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 7: RV32F (fma/cmp/cvt)",       (r6.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 8: Memory Access (lb/sb/lh/sh)",(r7.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 8: vecmul (guest)",        (r8.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 9: scal_mul (guest)",      (r9.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 10: gelu (guest)",         (r10.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 11: softmax (guest)",      (r11.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 12: softmax_norm (guest)", (r12.errors == 0) ? "PASS" : "FAIL");
    printf("  %-45s %s\n", "Test 14: conv2d (guest)",       (r14.errors == 0) ? "PASS" : "FAIL");
    printf("Total errors: %d\n", total_errors);
    printf("===========================================\n");

    free(s.vram_ptr);
    return total_errors ? 1 : 0;
}
