/*
 * tcu.h — TCU Tensor Accelerator Interface
 * Copyright (c) 2024-2025  Licensed under GPL v2 or later.
 */
#ifndef TCU_TCU_H
#define TCU_TCU_H

#include "../state.h"

int tcu_matmul(GPGPUState *s, uint32_t M, uint32_t K, uint32_t N, uint32_t a_base, uint32_t b_base, uint32_t c_base);

#endif
