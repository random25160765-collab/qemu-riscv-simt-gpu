/*
 * scheduler.h — GPU Kernel Scheduler Interface
 *
 * 管理 grid→block→warp→lane 调度，预译码缓存，SoA 编组。
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef GPGPU_SCHEDULER_H
#define GPGPU_SCHEDULER_H

#include <stdint.h>
#include "state.h"

/*
 * ============================================================================
 * 核心 API
 * ============================================================================
 *
 * 执行一个 kernel launch (根据 s->kernel 中的参数)。
 * 返回 0=成功, -1=错误。
 */
int scheduler_run_kernel(GPGPUState *s);

#endif /* GPGPU_SCHEDULER_H */
