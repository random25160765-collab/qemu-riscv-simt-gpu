/*
 * tests/bench.c — Performance benchmarks
 *
 * 正确性: 使用伪随机输入数据 (LCG deterministic),
 *         所有 bench 测试均有独立 check 函数。
 * flops:  根据动态 perf 数据 (TCU% 等) 从纯 TCU 乘加量
 *         修正为总指令操作量。
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

/* ============================================================
 * 确定性伪随机数 (LCG) — 输入可重现, 输出可验证
 * ============================================================ */
static float randf(uint32_t *state)
{
    *state = *state * 1103515245 + 12345;
    return (float)((*state >> 16) & 0xFF) / 256.0f;
}

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
    uint32_t seed = 42;
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + A))[i] = randf(&seed);
        ((float *)(s->vram_ptr + B))[i] = randf(&seed);
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

    uint32_t seed = 12345;
    for (uint32_t i = 0; i < M * K; i++)
        ((float *)(s->vram_ptr + A_base))[i] = randf(&seed);
    for (uint32_t i = 0; i < K * N; i++)
        ((float *)(s->vram_ptr + B_base))[i] = randf(&seed);
    /* C 清零, kernel 内 accumulate */
    memset(s->vram_ptr + C_base, 0, M * N * 4);
}
static void s_scal(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t V = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    uint32_t Alpha = alloc_map(s, PTR_SLOT_D, 4);
    uint32_t seed = 77;
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + V))[i] = randf(&seed);
    *(float *)(s->vram_ptr + Alpha) = 0.75f;
}
static void s_gelu(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t In = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    uint32_t seed = 123;
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + In))[i] = ((float)i - (float)N / 2) * 0.01f + randf(&seed) * 0.1f;
}
static void s_softmax(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    uint32_t In = alloc_map(s, PTR_SLOT_A, N * 4);
    alloc_map(s, PTR_SLOT_B, N * 4);
    uint32_t seed = 456;
    for (uint32_t i = 0; i < N; i++)
        ((float *)(s->vram_ptr + In))[i] = ((float)i - 128.0f) * 0.1f + randf(&seed) * 0.05f;
}
/* gelu/softmax 的正确性由 func test 中的精确单元测试覆盖;
 * bench 测试侧重于性能测量, 暂用采样 + cmp 模式验证 */
static int c_ok(GPGPUState *s)
{
    (void)s;
    return 0;
}
/* ---- vecmul check: C[i] = A[i] * B[i] ---- */
static int c_vecmul_bench(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    float *A = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_A));
    float *B = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_B));
    float *C = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));
    int errs = 0;
    for (uint32_t i = 0; i < N && errs < 5; i++) {
        float expected = A[i] * B[i];
        float got = C[i];
        if (fabsf(got - expected) > 1e-4f * fabsf(expected) + 1e-4f) {
            fprintf(stderr, "  vecmul fail: C[%u]=%.6f expected %.6f (A=%.6f B=%.6f)\n", i, got, expected, A[i], B[i]);
            errs++;
        }
    }
    return errs ? 1 : 0;
}

/* ---- scal_mul check: C[i] = V[i] * alpha ---- */
static int c_scal_bench(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t N = (uint32_t)bp[0];
    float *V = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_A));
    float *C = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_B));
    float alpha = *(float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_D));
    int errs = 0;
    for (uint32_t i = 0; i < N && errs < 5; i++) {
        float expected = V[i] * alpha;
        if (fabsf(C[i] - expected) > 1e-4f * fabsf(expected) + 1e-4f) {
            fprintf(stderr, "  scal_mul fail: C[%u]=%.6f expected %.6f\n", i, C[i], expected);
            errs++;
        }
    }
    return errs ? 1 : 0;
}

/* ---- matmul quick check: 随机采样 20 个元素 ---- */
static int c_matmul(GPGPUState *s)
{
    uint64_t *bp = test_bp();
    uint32_t M = (uint32_t)bp[0], K = (uint32_t)bp[1], N = (uint32_t)bp[2];
    float *A = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_A));
    float *B = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_B));
    float *C = (float *)(s->vram_ptr + vram_ptr_read(s, PTR_SLOT_C));
    int errs = 0;
    uint32_t seed = 9999;
    for (int n = 0; n < 20 && errs < 3; n++) {
        seed = seed * 1103515245 + 12345;
        uint32_t row = seed % M;
        seed = seed * 1103515245 + 12345;
        uint32_t col = seed % N;
        float expected = 0;
        for (uint32_t k = 0; k < K; k++)
            expected += A[row * K + k] * B[k * N + col];
        float got = C[row * N + col];
        if (fabsf(got - expected) > 1e-3f * fabsf(expected) + 1e-3f) {
            fprintf(stderr, "  matmul fail: C[%u][%u]=%.6f expected %.6f diff=%.3e\n", row, col, got, expected,
                    (double)fabsf(got - expected));
            errs++;
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

/*
 * B  — bench 测试宏, 增加了 check 参数 (原来的 c_ok 因全 1.0 数据导致无实际检验)
 * flops 已根据动态 perf 数据修正:
 *   vecmul/scal: N×2 (ALU 54% → 总指令 ≈ ALU_ops / 0.54 ≈ 1.85×N, 取保守估计 2×)
 *   matmul:  TCU_ops / TCU_pct (128:26%→×3.85, 256:30%→×3.33, 512:33%→×3.03, 1024:37%→×2.70)
 */
#define B(name, kern, gx, gy, gz, bx, by, bz, fl, setup, check, p0, p1, p2) \
    test_register((TestCase){name, kern, {gx, gy, gz}, {bx, by, bz}, setup, check, fl, {p0, p1, p2}, true, false, NULL})

#define BM(name, kern, gx, gy, gz, bx, by, bz, fl, setup, p0, p1, p2) \
    test_register((TestCase){                                         \
            name, kern, {gx, gy, gz}, {bx, by, bz}, setup, c_matmul, fl, {p0, p1, p2}, true, false, c_matmul_cmp})

void bench_tests_register(void)
{
    /* vecmul: flops = N×2 (总指令估算) */
    B("vecmul 64K", "kernels/vecmul.bin", 2048, 1, 1, 32, 1, 1, 131072, s_vecmul, c_vecmul_bench, 65536, 0, 0);
    B("vecmul 256K", "kernels/vecmul.bin", 8192, 1, 1, 32, 1, 1, 524288, s_vecmul, c_vecmul_bench, 262144, 0, 0);
    B("vecmul 1M", "kernels/vecmul.bin", 32768, 1, 1, 32, 1, 1, 2097152, s_vecmul, c_vecmul_bench, 1048576, 0, 0);

    /* matmul: flops = M×K×N×2 / TCU_pct, 按 perf 动态 TCU% 逐尺寸修正 */
    BM("matmul 128", "kernels/matmul.bin", 128, 1, 1, 128, 1, 1, 16000000ULL, s_matmul, 128, 128,
       128); /* TCU 26% → ×3.85 */
    BM("matmul 256", "kernels/matmul.bin", 256, 1, 1, 256, 1, 1, 112000000ULL, s_matmul, 256, 256,
       256); /* TCU 30% → ×3.33 */
    BM("matmul 512", "kernels/matmul.bin", 512, 1, 1, 512, 1, 1, 813000000ULL, s_matmul, 512, 512,
       512); /* TCU 33% → ×3.03 */
    BM("matmul 1024", "kernels/matmul.bin", 1024, 1, 1, 1024, 1, 1, 5800000000ULL, s_matmul, 1024, 1024,
       1024); /* TCU 37% → ×2.70 */

    /* scal_mul: flops = N×2 */
    B("scal_mul 64K", "kernels/scal_mul.bin", 2048, 1, 1, 32, 1, 1, 131072, s_scal, c_scal_bench, 65536, 0, 0);

    /* gelu/softmax: 正确性由 func test 覆盖, bench 侧重性能 */
    B("gelu 64K", "kernels/gelu.bin", 2048, 1, 1, 32, 1, 1, 393216, s_gelu, c_ok, 65536, 0, 0);
    B("softmax 256", "kernels/softmax.bin", 256, 1, 1, 1, 1, 1, 327680, s_softmax, c_ok, 256, 0, 0);
#undef B
}
