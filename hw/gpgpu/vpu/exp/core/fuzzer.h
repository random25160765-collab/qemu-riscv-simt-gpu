/*
 * fuzzer.h — VPU Engine Fuzzing Framework
 *
 * 随机生成指令序列，多 active_mask 交叉验证，检测 engine 不变量。
 * Copyright (c) 2024-2025, GPL v2.
 */
#ifndef VPU_FUZZER_H
#define VPU_FUZZER_H

#include <stdint.h>
#include "state.h"

/*
 * 运行 fuzzer: iterations 轮随机测试。
 * 返回 0=全部通过，非零=发现的错误数。
 */
int fuzzer_run(GPGPUState *s, uint32_t seed, int iterations, int verbose);

#endif
