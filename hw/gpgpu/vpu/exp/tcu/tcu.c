/*
 * tcu.c — TCU Tensor Accelerator
 * Host SIMD matmul. Called from engine.c TCU handlers.
 * Copyright (c) 2024-2025  Licensed under GPL v2 or later.
 */
#include <stdint.h>
#include "tcu.h"

int tcu_matmul(GPGPUState *s, uint32_t M, uint32_t K, uint32_t N, uint32_t a_base, uint32_t b_base, uint32_t c_base)
{
    float *a = (float *)(s->vram_ptr + a_base);
    float *b = (float *)(s->vram_ptr + b_base);
    float *c = (float *)(s->vram_ptr + c_base);
    for (uint32_t row = 0; row < M; row++) {
        for (uint32_t k = 0; k < K; k++) {
            float aik = a[row * K + k];
            float *c_row = c + row * N;
            float *b_row = b + k * N;
            for (uint32_t col = 0; col < N; col++)
                c_row[col] += aik * b_row[col];
        }
    }
    return 0;
}
