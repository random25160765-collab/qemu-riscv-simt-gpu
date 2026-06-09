/*
 * engine_types.h — 引擎类型定义 (SIMTFrame + EngineContext)
 *
 * 从 engine.h 拆分, 避免引入 predecode.h → dispatch.h 的强耦合链。
 * engine.h 和 engine_mod.c 均可 include 此文件。
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef GPGPU_ENGINE_TYPES_H
#define GPGPU_ENGINE_TYPES_H

#include <stdint.h>
#include "state.h"

/* SIMT stack frame (exposed for WarpSlot persistence) */
typedef struct {
    int32_t ft_idx;
    uint32_t mask;
} SIMTFrame;

/*
 * ============================================================================
 * 引擎上下文 — 引擎可见的最小状态
 * ============================================================================
 */
typedef struct {
    GPGPUState *s;     /* VRAM 访问 + kernel 参数 (只读) */
    uint32_t active;   /* 活跃掩码：0x1=标量, 0xFFFFFFFF=全 warp */
    uint8_t *shm;      /* shared memory buffer (NULL if none) */
    uint32_t shm_size; /* shared memory size */

    /* per-warp SIMT 上下文 (线程安全: 每个 warp 一份, 不共享) */
    uint32_t thread_id[3]; /* 当前线程 ID (base + lane offset) */
    uint32_t block_id[3];  /* 所属 block ID */
    uint32_t warp_id;      /* warp 编号 */
    uint32_t thread_mask;  /* 活跃线程掩码 */
    uint32_t vl;           /* 向量长度 (vsetvli 设置, 默认 32) */
} EngineContext;

#endif /* GPGPU_ENGINE_TYPES_H */
