/*
 * tests/bench.c — Performance benchmarks
 */
#include <stdint.h>
#include <string.h>
#include "../state.h"
#include "../test_runner.h"

extern uint64_t *test_bp(void);

/* --- bench setups --- */
static void s_vecmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + 0x200000))[i] = 3.0f;
    }
}
static void s_matmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t M = (uint32_t)bp[0], K = (uint32_t)bp[1], N = (uint32_t)bp[2];
    *(uint32_t *)s->vram_ptr = K;
    for (uint32_t i = 0; i < M * K; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = 1.0f;
    for (uint32_t i = 0; i < K * N; i++)
        ((float *)(s->vram_ptr + 0x200000))[i] = 1.0f;
}
static void s_scal(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
    *(float *)(s->vram_ptr + 0x400000) = 0.75f;
}
static void s_gelu(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)i * 0.01f;
}
static void s_softmax(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = ((float)i - 128.0f) * 0.1f;
}
static int c_ok(GPGPUState *s)
{
    (void)s;
    return 0;
}

#define B(name, kern, gx, gy, gz, bx, by, bz, fl, setup, p0, p1, p2) \
    test_register((TestCase){name, kern, {gx, gy, gz}, {bx, by, bz}, setup, c_ok, fl, {p0, p1, p2}, true, false})

void bench_tests_register(void)
{
    B("vecmul 64K", "kernels/vecmul.bin", 2048, 1, 1, 32, 1, 1, 65536, s_vecmul, 65536, 0, 0);
    B("vecmul 256K", "kernels/vecmul.bin", 8192, 1, 1, 32, 1, 1, 262144, s_vecmul, 262144, 0, 0);
    B("vecmul 1M", "kernels/vecmul.bin", 32768, 1, 1, 32, 1, 1, 1048576, s_vecmul, 1048576, 0, 0);
    B("matmul 128", "kernels/matmul.bin", 128, 1, 1, 128, 1, 1, 4194304ULL, s_matmul, 128, 128, 128);
    B("matmul 256", "kernels/matmul.bin", 256, 1, 1, 256, 1, 1, 33554432ULL, s_matmul, 256, 256, 256);
    B("matmul 512", "kernels/matmul.bin", 512, 1, 1, 512, 1, 1, 268435456ULL, s_matmul, 512, 512, 512);
    B("matmul 1024", "kernels/matmul.bin", 1024, 1, 1, 1024, 1, 1, 2147483648ULL, s_matmul, 1024, 1024, 1024);
    B("scal_mul 64K", "kernels/scal_mul.bin", 2048, 1, 1, 32, 1, 1, 65536, s_scal, 65536, 0, 0);
    B("gelu 64K", "kernels/gelu.bin", 2048, 1, 1, 32, 1, 1, 393216, s_gelu, 65536, 0, 0);
    B("softmax 256", "kernels/softmax.bin", 256, 1, 1, 1, 1, 1, 327680, s_softmax, 256, 0, 0);
#undef B
}
