/*
 * tests/native.c — Native C comparison (interpreter vs hand-written C)
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "../state.h"
#include "../core/scheduler.h"

#define KERN_ADDR 0x500000
extern void load_kernel(const char *path, GPGPUState *s, uint32_t addr);

static void run_native_vecmul(GPGPUState *s)
{
    uint32_t N = 65536, nw = N / 32;
    struct timespec T0, T1;
    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + 0x100000))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + 0x200000))[i] = 3.0f;
    }
    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0] = nw;
    s->kernel.grid_dim[1] = 1;
    s->kernel.grid_dim[2] = 1;
    s->kernel.block_dim[0] = 1;
    s->kernel.block_dim[1] = 1;
    s->kernel.block_dim[2] = 1;
    load_kernel("kernels/vecmul.bin", s, KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double ti = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;

    volatile float sum = 0;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    for (uint32_t w = 0; w < nw; w++) {
        for (int l = 0; l < 32; l++) {
            uint32_t t = w * 32 + l;
            float a = *(float *)(s->vram_ptr + 0x100000 + t * 4);
            float b = *(float *)(s->vram_ptr + 0x200000 + t * 4);
            *(float *)(s->vram_ptr + 0x300000 + t * 4) = a * b;
            sum += a * b;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double tn = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;
    printf("  Interp: %5.0fus  Native: %5.0fus  Ratio: %.1fx\n", ti * 1e6, tn * 1e6, ti / tn);
    (void)sum;
}

static void run_native_matmul(GPGPUState *s)
{
    int M = 128, K = 128, N = 128;
    struct timespec T0, T1;
    *(uint32_t *)s->vram_ptr = K;
    for (int i = 0; i < M * K; i++)
        ((float *)(s->vram_ptr + 0x100000))[i] = 1.0f;
    for (int i = 0; i < K * N; i++)
        ((float *)(s->vram_ptr + 0x600000))[i] = 1.0f;
    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0] = M;
    s->kernel.grid_dim[1] = 1;
    s->kernel.grid_dim[2] = 1;
    s->kernel.block_dim[0] = N;
    s->kernel.block_dim[1] = 1;
    s->kernel.block_dim[2] = 1;
    load_kernel("kernels/matmul.bin", s, KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double ti = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;

    volatile float sum = 0;
    float *A = (float *)(s->vram_ptr + 0x100000), *B = (float *)(s->vram_ptr + 0x600000);
    float *C = (float *)(s->vram_ptr + 0xB00000);
    memset(C, 0, M * N * 4);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    for (int row = 0; row < M; row++)
        for (int k = 0; k < K; k++) {
            float aik = A[row * K + k];
            float *cr = C + row * N, *br = B + k * N;
            for (int col = 0; col < N; col++)
                cr[col] += aik * br[col];
        }
    for (int i = 0; i < M * N; i++)
        sum += C[i];
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double tn = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;
    printf("  Interp: %5.0fus  Native: %5.0fus  Ratio: %.1fx\n", ti * 1e6, tn * 1e6, ti / tn);
    (void)sum;
}

void test_run_native(GPGPUState *s)
{
    printf("=== Native C Comparison ===\n\n");
    printf("--- vecmul ---\n");
    run_native_vecmul(s);
    printf("--- matmul ---\n");
    run_native_matmul(s);
    printf("\n");
}
