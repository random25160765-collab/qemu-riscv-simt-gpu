/*
 * vpu.h — VPU Vector Accelerator Interface
 * Copyright (c) 2024-2025  Licensed under GPL v2 or later.
 */
#ifndef VPU_VPU_H
#define VPU_VPU_H

#include "../state.h"

int vpu_vecmul(GPGPUState *s, uint32_t n, uint32_t a_base, uint32_t b_base, uint32_t c_base);
int vpu_scal_mul(GPGPUState *s, uint32_t n, float alpha, uint32_t v_base, uint32_t c_base);
int vpu_saxpy(GPGPUState *s, uint32_t n, float alpha, uint32_t x_base, uint32_t y_base);
int vpu_gelu(GPGPUState *s, uint32_t n, uint32_t i_base, uint32_t o_base);
int vpu_softmax(GPGPUState *s, uint32_t n, uint32_t i_base, uint32_t o_base);
int vpu_memcpy(GPGPUState *s, uint32_t n, uint32_t src, uint32_t dst);

#endif
