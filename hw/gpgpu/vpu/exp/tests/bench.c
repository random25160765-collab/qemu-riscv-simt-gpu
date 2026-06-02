/*
 * tests/bench.c — Performance benchmarks
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../state.h"
#include "../test_runner.h"
#include "../core/vram_alloc.h"

extern uint64_t *test_bp(void);

static uint32_t alloc_map(GPGPUState *s, int slot, size_t bytes)
{
    uint32_t addr = vram_alloc(s, bytes);
    vram_ptr_write(s, slot, addr);
    return addr;
}

/* --- bench setups --- */
static void s_vecmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t A = alloc_map(s, PTR_SLOT_A, N * 4);
    uint32_t B = alloc_map(s, PTR_SLOT_B, N * 4);
    alloc_map(s, PTR_SLOT_C, N * 4);
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + A))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + B))[i] = 3.0f;
    }
}
static void s_matmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t M = (uint32_t)bp[0], K = (uint32_t)bp[1], N = (uint32_t)bp[2];
    *(uint32_t *)s->vram_ptr = K;

    uint32_t A_base = alloc_map(s, PTR_SLOT_A, M * K * 4);
    uint32_t B_base = alloc_map(s, PTR_SLOT_B, K * N * 4);
    uint32_t C_base = alloc_map(s, PTR_SLOT_C, M * N * 4);

    for (uint32_t i = 0; i < M * K; i++)
        ((float *)(s->vram_ptr + A_base))[i] = 1.0f;
    for (uint32_t i = 0; i < K * N; i++)
        ((float *)(s->vram_ptr + B_base))[i] = 1.0f;
}
static void s_scal(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t V = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    uint32_t Alpha = alloc_map(s, PTR_SLOT_D, 4);
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + V))[i] = (float)(i % 256);
    *(float *)(s->vram_ptr + Alpha) = 0.75f;
}
static void s_gelu(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t In = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + In))[i] = (float)i * 0.01f;
}
static void s_softmax(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t In = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + In))[i] = ((float)i - 128.0f) * 0.1f;
}
static int c_ok(GPGPUState *s)
{
    (void)s;
    return 0;
}

static int c_matmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t M = (uint32_t)bp[0], K = (uint32_t)bp[1], N = (uint32_t)bp[2];
    float *C = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));
    int errs = 0;
    for (uint32_t row = 0; row < M && errs < 5; row++) {
        for (uint32_t col = 0; col < N && errs < 5; col++) {
            float expected = (float)K; /* all A=B=1.0, so C[i][j]=K */
            float got = C[row * N + col];
            if (fabsf(got - expected) > 1e-4f * expected + 1e-4f) {
                fprintf(stderr, "  matmul fail: C[%u][%u]=%.3f expected %.3f\n", row, col, got, expected);
                errs++;
            }
        }
    }
    return errs ? 1 : 0;
}

/* Native host matmul, then compare against interpreter output */
static int c_matmul_cmp(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t M = (uint32_t)bp[0], K = (uint32_t)bp[1], N = (uint32_t)bp[2];
    float *A = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_A));
    float *B = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_B));
    float *C_interp = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));

    float *C_native = calloc(M * N, sizeof(float));
    if (!C_native) {
        fprintf(stderr, "  cmp: OOM\n");
        return 1;
    }

    /* host matmul: C = A × B */
    for (uint32_t row = 0; row < M; row++)
        for (uint32_t k = 0; k < K; k++) {
            float aik = A[row * K + k];
            float *cr = C_native + row * N;
            float *br = B + k * N;
            for (uint32_t col = 0; col < N; col++)
                cr[col] += aik * br[col];
        }

    int errs = 0;
    for (uint32_t i = 0; i < M * N && errs < 10; i++) {
        float got = C_interp[i], exp = C_native[i];
        if (fabsf(got - exp) > 1e-4f * fabsf(exp) + 1e-4f) {
            uint32_t row = i / N, col = i % N;
            fprintf(stderr, "  matmul cmp: C[%u][%u] interp=%.6f native=%.6f diff=%.6e\n", row, col, got, exp,
                    (double)fabsf(got - exp));
            errs++;
        }
    }
    free(C_native);
    return errs ? 1 : 0;
}

#define B(name, kern, gx, gy, gz, bx, by, bz, fl, setup, p0, p1, p2) \
    test_register((TestCase){name, kern, {gx, gy, gz}, {bx, by, bz}, setup, c_ok, fl, {p0, p1, p2}, true, false, NULL})

#define BM(name, kern, gx, gy, gz, bx, by, bz, fl, setup, p0, p1, p2) \
    test_register((TestCase){                                         \
            name, kern, {gx, gy, gz}, {bx, by, bz}, setup, c_matmul, fl, {p0, p1, p2}, true, false, c_matmul_cmp})

void bench_tests_register(void)
{
    B("vecmul 64K", "kernels/vecmul.bin", 2048, 1, 1, 32, 1, 1, 65536, s_vecmul, 65536, 0, 0);
    B("vecmul 256K", "kernels/vecmul.bin", 8192, 1, 1, 32, 1, 1, 262144, s_vecmul, 262144, 0, 0);
    B("vecmul 1M", "kernels/vecmul.bin", 32768, 1, 1, 32, 1, 1, 1048576, s_vecmul, 1048576, 0, 0);
    BM("matmul 128", "kernels/matmul.bin", 128, 1, 1, 128, 1, 1, 4194304ULL, s_matmul, 128, 128, 128);
    BM("matmul 256", "kernels/matmul.bin", 256, 1, 1, 256, 1, 1, 33554432ULL, s_matmul, 256, 256, 256);
    BM("matmul 512", "kernels/matmul.bin", 512, 1, 1, 512, 1, 1, 268435456ULL, s_matmul, 512, 512, 512);
    BM("matmul 1024", "kernels/matmul.bin", 1024, 1, 1, 1024, 1, 1, 2147483648ULL, s_matmul, 1024, 1024, 1024);
    B("scal_mul 64K", "kernels/scal_mul.bin", 2048, 1, 1, 32, 1, 1, 65536, s_scal, 65536, 0, 0);
    B("gelu 64K", "kernels/gelu.bin", 2048, 1, 1, 32, 1, 1, 393216, s_gelu, 65536, 0, 0);
    B("softmax 256", "kernels/softmax.bin", 256, 1, 1, 1, 1, 1, 327680, s_softmax, 256, 0, 0);
#undef B
}
