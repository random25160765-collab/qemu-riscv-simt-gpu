/*
 * test_runner.h — VPU Test Framework
 * Copyright (c) 2024-2025, GPL v2
 */
#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdint.h>
#include <stdbool.h>
#include "state.h"

typedef struct {
    const char *name, *kernel;
    uint32_t grid[3], block[3];
    void (*setup)(GPGPUState *s);
    int (*check)(GPGPUState *s);
    uint64_t flops, params[3];
    bool bench, native;
} TestCase;

void test_register(TestCase t);
int test_run(GPGPUState *s, const char *group, const char *filter);
void test_run_native(GPGPUState *s);
void test_run_all(GPGPUState *s);

#endif
