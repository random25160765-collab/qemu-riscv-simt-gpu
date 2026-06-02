/*
 * vpu.c — VPU Vector Accelerator
 * Host SIMD batch primitives. Called from engine.c VPU handlers.
 * Copyright (c) 2024-2025  Licensed under GPL v2 or later.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "vpu.h"
#include "../core/sfu.h"

int vpu_vecmul(GPGPUState *s, uint32_t n, uint32_t a_base, uint32_t b_base, uint32_t c_base)
{
    float *a = (float *)(s->vram_ptr + a_base);
    float *b = (float *)(s->vram_ptr + b_base);
    float *c = (float *)(s->vram_ptr + c_base);
    for (uint32_t i = 0; i < n; i++)
        c[i] = a[i] * b[i];
    return 0;
}

int vpu_scal_mul(GPGPUState *s, uint32_t n, float alpha, uint32_t v_base, uint32_t c_base)
{
    float *v = (float *)(s->vram_ptr + v_base);
    float *c = (float *)(s->vram_ptr + c_base);
    for (uint32_t i = 0; i < n; i++)
        c[i] = v[i] * alpha;
    return 0;
}

int vpu_saxpy(GPGPUState *s, uint32_t n, float alpha, uint32_t x_base, uint32_t y_base)
{
    float *x = (float *)(s->vram_ptr + x_base);
    float *y = (float *)(s->vram_ptr + y_base);
    for (uint32_t i = 0; i < n; i++)
        y[i] = alpha * x[i] + y[i];
    return 0;
}

int vpu_gelu(GPGPUState *s, uint32_t n, uint32_t i_base, uint32_t o_base)
{
    float *in = (float *)(s->vram_ptr + i_base);
    float *out = (float *)(s->vram_ptr + o_base);
    for (uint32_t i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x * fastsigmoid(1.702f * x);
    }
    return 0;
}

int vpu_softmax(GPGPUState *s, uint32_t n, uint32_t i_base, uint32_t o_base)
{
    float *in = (float *)(s->vram_ptr + i_base);
    float *out = (float *)(s->vram_ptr + o_base);
    float max = in[0];
    for (uint32_t i = 1; i < n; i++)
        if (in[i] > max) max = in[i];
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = expf(in[i] - max);
        sum += out[i];
    }
    float inv = 1.0f / sum;
    for (uint32_t i = 0; i < n; i++)
        out[i] *= inv;
    return 0;
}

int vpu_memcpy(GPGPUState *s, uint32_t n, uint32_t src, uint32_t dst)
{
    memcpy(s->vram_ptr + dst, s->vram_ptr + src, n);
    return 0;
}
