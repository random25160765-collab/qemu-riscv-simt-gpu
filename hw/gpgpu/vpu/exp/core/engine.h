/*
 * engine.h — RISC-V SIMT Execution Engine Interface
 *
 * 合并 threaded + simd，统一 SoA 数据布局。
 * active 掩码控制 lane 宽度：0x1=标量，0xFFFFFFFF=全 warp。
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef GPGPU_ENGINE_H
#define GPGPU_ENGINE_H

#include <stdint.h>
#include "state.h"
#include "predecode.h" /* ThOp */

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
} EngineContext;

/*
 * ============================================================================
 * 引擎入口
 * ============================================================================
 *
 * 执行 predecoded ThOp[] 直到 ebreak/done/error。
 *
 * 参数:
 *   code    — 预译码的 ThOp 数组 (handler 已解析为 computed-goto 标签)
 *   tcount  — 指令条数
 *   ctx     — 引擎上下文 (VRAM 指针 + active 掩码)
 *   gpr     — SoA 通用寄存器 [32 regs × 32 lanes]
 *   fpr     — SoA 浮点寄存器 [32 regs × 32 lanes]
 *   pc      — 32 个 lane 的程序计数器
 *
 * 返回:
 *   0  = 成功 (ebreak/done)
 *   -1 = 非法指令
 */
int engine_exec(ThOp *code, int tcount, const EngineContext *ctx, uint32_t gpr[32 * 32], uint32_t fpr[32 * 32],
                uint32_t pc[32], uint32_t mhartid[32], uint32_t fcsr[32], SIMTFrame *_stk, int *_sdepth, int resume_pc);

/*
 * ============================================================================
 * Handler 解析 — 调度器调用，将 instr_id → computed-goto 标签
 * ============================================================================
 */
void engine_resolve_handlers(ThOp *code, int tcount);

#endif /* GPGPU_ENGINE_H */
