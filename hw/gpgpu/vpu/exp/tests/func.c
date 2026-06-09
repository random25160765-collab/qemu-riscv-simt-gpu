/*
 * tests/func.c — Functional verification tests
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "../state.h"
#include "../test_runner.h"
#include "../core/vram_alloc.h"

/* helper: allocate + write ptr_table, return address */
static uint32_t alloc_map(GPGPUState *s, int slot, size_t bytes)
{
    uint32_t addr = vram_alloc(s, bytes);
    vram_ptr_write(s, slot, addr);
    return addr;
}

/* --- vecmul --- */
static void s_vm(GPGPUState *s)
{
    uint32_t A = alloc_map(s, PTR_SLOT_A, 2048 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 2048 * 4);
    alloc_map(s, PTR_SLOT_C, 2048 * 4);
    for (uint32_t i = 0; i < 2048; i++) {
        ((float *)(s->vram_ptr + A))[i] = (float)(i + 1);
        ((float *)(s->vram_ptr + B))[i] = 2.0f;
    }
}
static int c_vm(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    for (uint32_t i = 0; i < 2048; i++)
        if (fabsf(((float *)(s->vram_ptr + C))[i] - (float)(i + 1) * 2.0f) > 1e-5f) return 1;
    return 0;
}

/* --- saxpy --- */
static void s_saxpy(GPGPUState *s)
{
    uint32_t N = 16384;
    srand48(42);
    *(uint32_t *)s->vram_ptr = N;
    *(float *)(s->vram_ptr + 4) = 2.5f;
    uint32_t X = alloc_map(s, PTR_SLOT_A, N * 4);
    uint32_t Y = alloc_map(s, PTR_SLOT_B, N * 4);
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + X))[i] = (float)(drand48() * 100 - 50);
        ((float *)(s->vram_ptr + Y))[i] = (float)(drand48() * 100 - 50);
    }
}
static int c_saxpy(GPGPUState *s)
{
    uint32_t N = 16384;
    float A = 2.5f;
    srand48(42);
    float *y = malloc(N * 4);
    uint32_t X = vram_ptr_read(s, PTR_SLOT_A);
    uint32_t Y = vram_ptr_read(s, PTR_SLOT_B);
    for (uint32_t i = 0; i < N; i++) {
        drand48();
        y[i] = (float)(drand48() * 100 - 50);
    }
    int e = 0;
    for (uint32_t i = 0; i < N && e < 10; i++) {
        float exp = A * ((float *)(s->vram_ptr + X))[i] + y[i];
        if (fabsf(((float *)(s->vram_ptr + Y))[i] - exp) > 1e-4f * fabsf(exp)) e++;
    }
    free(y);
    return e ? 1 : 0;
}

/* --- rv32m --- */
static void s_rv32m(GPGPUState *s)
{
    uint32_t cs[][2] = {{10, 3},  {10, (uint32_t)-3}, {(uint32_t)-5, (uint32_t)-2},
                        {100, 0}, {0x80000000, 2},    {0xFFFFFFFF, 0xFFFFFFFF}};
    *(uint32_t *)s->vram_ptr = 6;
    uint32_t In = alloc_map(s, PTR_SLOT_A, sizeof(cs));
    alloc_map(s, PTR_SLOT_C, 6 * 32);
    memcpy(s->vram_ptr + In, cs, sizeof(cs));
}
static int c_rv32m(GPGPUState *s)
{
    uint32_t cs[][2] = {{10, 3},  {10, (uint32_t)-3}, {(uint32_t)-5, (uint32_t)-2},
                        {100, 0}, {0x80000000, 2},    {0xFFFFFFFF, 0xFFFFFFFF}};
    uint32_t Out = vram_ptr_read(s, PTR_SLOT_C);
    for (uint32_t i = 0; i < 6; i++) {
        int32_t a = cs[i][0], b = cs[i][1];
        uint32_t *o = (uint32_t *)(s->vram_ptr + Out) + i * 8;
        uint32_t ex[] = {
                (uint32_t)(a * b),
                (uint32_t)(((int64_t)a * (int64_t)b) >> 32),
                (uint32_t)(((int64_t)a * (uint64_t)(uint32_t)b) >> 32),
                (uint32_t)(((uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b) >> 32),
                b ? (uint32_t)(a / b) : (uint32_t)-1,
                b ? ((uint32_t)a / (uint32_t)b) : 0xFFFFFFFF,
                b ? (uint32_t)(a % b) : (uint32_t)a,
                b ? ((uint32_t)a % (uint32_t)b) : (uint32_t)a,
        };
        for (int j = 0; j < 8; j++)
            if (o[j] != ex[j]) return 1;
    }
    return 0;
}

/* --- rv32f --- */
static void s_rv32f(GPGPUState *s)
{
    float fa = 3, fb = 2, fc = 4;
    int32_t i32 = -5;
    uint32_t u32 = 7;
    uint32_t In = alloc_map(s, PTR_SLOT_A, 20);
    alloc_map(s, PTR_SLOT_C, 26 * 4);
    memcpy(s->vram_ptr + In, &fa, 4);
    memcpy(s->vram_ptr + In + 4, &fb, 4);
    memcpy(s->vram_ptr + In + 8, &fc, 4);
    memcpy(s->vram_ptr + In + 12, &i32, 4);
    memcpy(s->vram_ptr + In + 16, &u32, 4);
}
static int c_rv32f(GPGPUState *s)
{
    uint32_t Out = vram_ptr_read(s, PTR_SLOT_C);
    float *fo = (float *)(s->vram_ptr + Out);
    uint32_t *io = (uint32_t *)(s->vram_ptr + Out);
    float fa = 3, fb = 2, fc = 4;
    int e = 0;
#define CF(i, v) \
    if (fabsf(fo[i] - (float)(v)) > 1e-4f * fabsf((float)(v)) && fabsf(fo[i] - (float)(v)) > 1e-5f) e++
#define CI(i, v) \
    if (io[i] != (uint32_t)(v)) e++
    CF(0, fa + fb);
    CF(1, fa - fb);
    CF(2, fa * fb);
    CF(3, fa / fb);
    CF(4, sqrtf(fa));
    CF(5, fa * fb + fc);
    CF(6, fa * fb - fc);
    CF(7, -(fa * fb - fc));
    CF(8, -(fa * fb + fc));
    CF(9, 3);
    CF(10, -3);
    CF(11, 3);
    CF(12, 2);
    CF(13, 3);
    CI(14, 0);
    CI(15, 0);
    CI(16, 1);
    CI(17, 1);
    CI(18, 3);
    CI(19, 3);
    CF(20, -5);
    CF(21, 7);
    CI(22, 7);
    CI(23, 0x40400000);
    CI(24, 0x40);
    CI(25, 0x02);
#undef CF
#undef CI
    return e ? 1 : 0;
}

/* --- mem access --- */
static void s_mem(GPGPUState *s)
{
    uint8_t in[8] = {0x7F, 0x80, 0xFF, 0x00, 0x34, 0x12, 0x78, 0x56};
    uint32_t In = alloc_map(s, PTR_SLOT_A, 8);
    alloc_map(s, PTR_SLOT_C, 12 * 4);
    memcpy(s->vram_ptr + In, in, 8);
}
static int c_mem(GPGPUState *s)
{
    uint32_t *o = (uint32_t *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));
    return (o[0] != 127 || o[1] != (uint32_t)(int32_t)-128 || o[2] != (uint32_t)-1 || o[3] != 128 || o[4] != 255 ||
            o[5] != 0x1234 || o[6] != 0x5678 || o[7] != 0x1234 || o[8] != 0x00FF807F || o[9] != 0x42 ||
            o[10] != 0x4321 || o[11] != 0x00FF807F);
}

/* --- matmul --- */
static void s_mm(GPGPUState *s)
{
    *(uint32_t *)s->vram_ptr = 2;
    uint32_t A = alloc_map(s, PTR_SLOT_A, 2 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 2 * 4);
    uint32_t C = alloc_map(s, PTR_SLOT_C, 1 * 4);
    ((float *)(s->vram_ptr + A))[0] = 1;
    ((float *)(s->vram_ptr + A))[1] = 2;
    ((float *)(s->vram_ptr + B))[0] = 3;
    ((float *)(s->vram_ptr + B))[1] = 4;
}
static int c_mm(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    return fabsf(((float *)(s->vram_ptr + C))[0] - 11) > 1e-3f;
}

/* --- dot product --- */
static void s_dot(GPGPUState *s)
{
    uint32_t N = 32768;
    *(uint32_t *)s->vram_ptr = N;
    uint32_t A = alloc_map(s, PTR_SLOT_A, N * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, N * 4);
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + A))[i] = (float)(i % 100) * 0.01f;
        ((float *)(s->vram_ptr + B))[i] = (float)((i + 1) % 100) * 0.01f;
    }
}
static int c_dot(GPGPUState *s)
{
    float sum = 0;
    for (uint32_t i = 0; i < 32768; i++)
        sum += (float)(i % 100) * 0.01f * (float)((i + 1) % 100) * 0.01f;
    return fabsf(*(float *)(s->vram_ptr + 4) - sum) > 0.01f;
}

/* --- memcpy --- */
static void s_mcpy(GPGPUState *s)
{
    uint32_t N = 131072;
    *(uint32_t *)s->vram_ptr = N;
    uint32_t Src = alloc_map(s, PTR_SLOT_A, N * 4);
    uint32_t Dst = alloc_map(s, PTR_SLOT_B, N * 4);
    for (uint32_t i = 0; i < N; i++) {
        ((uint32_t *)(s->vram_ptr + Src))[i] = i;
        ((uint32_t *)(s->vram_ptr + Dst))[i] = 0;
    }
}
static int c_mcpy(GPGPUState *s)
{
    uint32_t Dst = vram_ptr_read(s, PTR_SLOT_B);
    for (uint32_t i = 0; i < 131072; i++)
        if (((uint32_t *)(s->vram_ptr + Dst))[i] != i) return 1;
    return 0;
}

/* --- simple tests --- */
static void s_v1(GPGPUState *s)
{
    uint32_t A = alloc_map(s, PTR_SLOT_A, 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 4);
    alloc_map(s, PTR_SLOT_C, 4);
    ((float *)(s->vram_ptr + A))[0] = 2.5f;
    ((float *)(s->vram_ptr + B))[0] = 3.0f;
}
static int c_v1(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    return fabsf(((float *)(s->vram_ptr + C))[0] - 7.5f) > 1e-5f;
}
static void s_s1(GPGPUState *s)
{
    uint32_t V = alloc_map(s, PTR_SLOT_A, 4);
    alloc_map(s, PTR_SLOT_B, 4);
    uint32_t Alpha = alloc_map(s, PTR_SLOT_D, 4);
    ((float *)(s->vram_ptr + V))[0] = 4.0f;
    *(float *)(s->vram_ptr + Alpha) = 0.75f;
}
static int c_s1(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_B);
    return fabsf(((float *)(s->vram_ptr + C))[0] - 3.0f) > 1e-5f;
}
static void s_gelu(GPGPUState *s)
{
    uint32_t In = alloc_map(s, PTR_SLOT_A, 4);
    alloc_map(s, PTR_SLOT_B, 4);
    ((float *)(s->vram_ptr + In))[0] = 0.5f;
}
static int c_gelu(GPGPUState *s)
{
    uint32_t Out = vram_ptr_read(s, PTR_SLOT_B);
    float expected = 0.5f * (1.0f / (1.0f + expf(-1.702f * 0.5f)));
    float actual = ((float *)(s->vram_ptr + Out))[0];
    return fabsf(actual - expected) > 1e-3f;
}
static void s_smax(GPGPUState *s)
{
    uint32_t In = alloc_map(s, PTR_SLOT_A, 64 * 4);
    alloc_map(s, PTR_SLOT_B, 64 * 4);
    for (uint32_t i = 0; i < 64; i++)
        ((float *)(s->vram_ptr + In))[i] = ((float)i - 32.0f) * 0.1f;
}
static int c_smax(GPGPUState *s)
{
    uint32_t Out = vram_ptr_read(s, PTR_SLOT_B);
    float sum = 0;
    for (uint32_t i = 0; i < 64; i++)
        sum += ((float *)(s->vram_ptr + Out))[i];
    return fabsf(sum - 1.0f) > 0.02f;
}
static void s_snorm(GPGPUState *s)
{
    uint32_t Tmp = alloc_map(s, PTR_SLOT_B, 4); /* kernel reads from slot B */
    alloc_map(s, PTR_SLOT_C, 4);                /* kernel writes to slot C */
    uint32_t Sum = alloc_map(s, PTR_SLOT_D, 4); /* kernel reads from slot D */
    ((float *)(s->vram_ptr + Tmp))[0] = expf(1.0f);
    *(float *)(s->vram_ptr + Sum) = expf(1.0f);
}
static int c_snorm(GPGPUState *s)
{
    uint32_t Out = vram_ptr_read(s, PTR_SLOT_C);
    return fabsf(((float *)(s->vram_ptr + Out))[0] - 1.0f) > 0.01f;
}
static void s_conv(GPGPUState *s)
{
    float in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, kr[] = {1, 0, 0, 1};
    uint32_t In = alloc_map(s, PTR_SLOT_A, 9 * 4);
    uint32_t Ker = alloc_map(s, PTR_SLOT_B, 4 * 4);
    alloc_map(s, PTR_SLOT_C, 4);
    for (int i = 0; i < 9; i++)
        ((float *)(s->vram_ptr + In))[i] = in[i];
    for (int i = 0; i < 4; i++)
        ((float *)(s->vram_ptr + Ker))[i] = kr[i];
}
static int c_conv(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    return fabsf(((float *)(s->vram_ptr + C))[0] - 6.0f) > 1e-4f;
}

/* --- VPU vecmul: 32 elements, one warp --- */
static void s_vpu_vm(GPGPUState *s)
{
    uint32_t A = alloc_map(s, PTR_SLOT_A, 32 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 32 * 4);
    alloc_map(s, PTR_SLOT_C, 32 * 4);
    for (int i = 0; i < 32; i++) {
        ((float *)(s->vram_ptr + A))[i] = (float)(i + 1);
        ((float *)(s->vram_ptr + B))[i] = 2.0f;
    }
}
static int c_vpu_vm(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    for (int i = 0; i < 32; i++)
        if (fabsf(((float *)(s->vram_ptr + C))[i] - (float)(i + 1) * 2.0f) > 1e-5f) return 1;
    return 0;
}

/* --- TCU single-warp matmul: M=1,K=4,N=32 --- */
static void s_mma_f(GPGPUState *s)
{
    *(uint32_t *)s->vram_ptr = 4; /* K */
    uint32_t A = alloc_map(s, PTR_SLOT_A, 4 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 4 * 32 * 4);
    alloc_map(s, PTR_SLOT_C, 32 * 4);
    for (int i = 0; i < 4; i++)
        ((float *)(s->vram_ptr + A))[i] = 1.0f;
    for (int i = 0; i < 4 * 32; i++)
        ((float *)(s->vram_ptr + B))[i] = 1.0f;
}
static int c_mma_f(GPGPUState *s)
{
    float *C = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));
    for (int i = 0; i < 32; i++)
        if (fabsf(C[i] - 4.0f) > 1e-4f) return 1;
    return 0;
}

/* --- RVV integer vector add: 32 × int32 --- */
static void s_vadd_int(GPGPUState *s)
{
    uint32_t A = alloc_map(s, PTR_SLOT_A, 32 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 32 * 4);
    alloc_map(s, PTR_SLOT_C, 32 * 4);
    for (int i = 0; i < 32; i++) {
        ((int32_t *)(s->vram_ptr + A))[i] = i;
        ((int32_t *)(s->vram_ptr + B))[i] = i * 2;
    }
}
static int c_vadd_int(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    for (int i = 0; i < 32; i++)
        if (((int32_t *)(s->vram_ptr + C))[i] != i * 3) return 1;
    return 0;
}

/* --- RVV VL=16 test: only first 16 elements processed --- */
static void s_vadd_vl(GPGPUState *s)
{
    uint32_t A = alloc_map(s, PTR_SLOT_A, 32 * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, 32 * 4);
    uint32_t C = alloc_map(s, PTR_SLOT_C, 32 * 4);
    for (int i = 0; i < 32; i++) {
        ((int32_t *)(s->vram_ptr + A))[i] = i;
        ((int32_t *)(s->vram_ptr + B))[i] = i * 2;
        ((int32_t *)(s->vram_ptr + C))[i] = -1; /* sentinel */
    }
}
static int c_vadd_vl(GPGPUState *s)
{
    uint32_t C = vram_ptr_read(s, PTR_SLOT_C);
    for (int i = 0; i < 16; i++)
        if (((int32_t *)(s->vram_ptr + C))[i] != i * 3) return 1;
    for (int i = 16; i < 32; i++)
        if (((int32_t *)(s->vram_ptr + C))[i] != -1) return 1; /* unchanged */
    return 0;
}

/* --- registration --- */
void func_tests_register(void)
{
#define F(name, kern, gx, gy, gz, bx, by, bz, setup, check) \
    test_register((TestCase){name, kern, {gx, gy, gz}, {bx, by, bz}, setup, check, 0, {0, 0, 0}, false, false, NULL})

    F("Vector Add", "kernels/vecmul.bin", 1, 1, 1, 2048, 1, 1, s_vm, c_vm);
    F("SAXPY", "kernels/saxpy.bin", 1, 1, 1, 1, 1, 1, s_saxpy, c_saxpy);
    F("RV32M", "kernels/rv32m.bin", 1, 1, 1, 1, 1, 1, s_rv32m, c_rv32m);
    F("RV32F", "kernels/rv32f.bin", 1, 1, 1, 1, 1, 1, s_rv32f, c_rv32f);
    F("Mem Access", "kernels/mem_access.bin", 1, 1, 1, 1, 1, 1, s_mem, c_mem);
    F("Matmul", "kernels/matmul.bin", 1, 1, 1, 1, 1, 1, s_mm, c_mm);
    F("Dot Product", "kernels/dot_product.bin", 1, 1, 1, 1, 1, 1, s_dot, c_dot);
    F("Memcpy", "kernels/memcpy.bin", 1, 1, 1, 1, 1, 1, s_mcpy, c_mcpy);
    F("Vecmul", "kernels/vecmul.bin", 1, 1, 1, 1, 1, 1, s_v1, c_v1);
    F("Scal_mul", "kernels/scal_mul.bin", 1, 1, 1, 1, 1, 1, s_s1, c_s1);
    F("GELU", "kernels/gelu.bin", 1, 1, 1, 1, 1, 1, s_gelu, c_gelu);
    F("Softmax", "kernels/softmax.bin", 1, 1, 1, 64, 1, 1, s_smax, c_smax);
    F("SoftmaxNorm", "kernels/softmax_norm.bin", 1, 1, 1, 1, 1, 1, s_snorm, c_snorm);
    F("Conv2d", "kernels/conv2d.bin", 3, 3, 1, 2, 1, 1, s_conv, c_conv);
    F("Vecmul VPU", "kernels/vecmul_vpu.bin", 1, 1, 1, 32, 1, 1, s_vpu_vm, c_vpu_vm);
    /* TCU single-warp: M=1,K=4,N=32, all-ones → C=4 */
    F("MMA 1x4x32", "kernels/matmul.bin", 1, 1, 1, 32, 1, 1, s_mma_f, c_mma_f);
    F("Vec Add Int", "kernels/vadd_int.bin", 1, 1, 1, 32, 1, 1, s_vadd_int, c_vadd_int);
    F("Vec Add VL=16", "kernels/vadd_vl.bin", 1, 1, 1, 32, 1, 1, s_vadd_vl, c_vadd_vl);

#undef F
}
