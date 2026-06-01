/*
 * test_runner.h — VPU Test Framework (registration-based)
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdint.h>
#include <stdbool.h>
#include "state.h"

typedef struct {
    const char *name;
    const char *kernel;
    uint32_t    grid[3];
    uint32_t    block[3];
    void      (*setup)(GPGPUState *s);
    int       (*check)(GPGPUState *s);
    uint64_t    flops;
    uint64_t    params[3];       /* bench params (M,K,N for matmul) */
    bool        bench;
    bool        native;
} TestCase;

void test_register(TestCase t);
int  test_run(GPGPUState *s, const char *group, const char *filter);
void test_run_native(GPGPUState *s);
void test_run_all(GPGPUState *s);

#endif
