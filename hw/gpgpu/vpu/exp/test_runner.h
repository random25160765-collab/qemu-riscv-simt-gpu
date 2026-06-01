/*
 * test_runner.h — VPU Interpreter Test Runner
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include "state.h"

/*
 * ============================================================================
 * 运行所有测试
 * ============================================================================
 *
 * 返回: 总错误数，0 = 全部通过
 */
int test_runner_run(GPGPUState *s);

#endif /* TEST_RUNNER_H */
