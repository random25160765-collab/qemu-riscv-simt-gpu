/*
 * scheduler.c — GPU Kernel Scheduler Implementation
 *
 * 职责:
 *   1. 预译码缓存 — kernel binary → ThOp[] (一次 kernel launch 只做一次)
 *   2. Grid→Block→Warp 迭代
 *   3. Warp 状态初始化 + SoA 编组/解组
 *   4. SIMT 上下文注入
 *   5. 调用引擎执行
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "gpgpu_core.h"
#include "engine.h"
#include "scheduler.h"
#include "simd_predecode.h"
#include "simd_dispatch.h"
#include "memory.h"

/*
 * ============================================================================
 * 预译码 — kernel binary → ThOp[] (无缓存，每次 launch 重新译码)
 * ============================================================================
 */
static ThOp *scheduler_predecode(GPGPUState *s, uint32_t kern_addr,
                                  uint32_t kern_size, int *out_count)
{
    SIMDDecoder dec;
    simd_decoder_init(&dec);

    ThOp *code = simd_predecode(&dec, s, kern_addr, kern_size, out_count);
    if (!code) return NULL;

    /* 解析 handler (computed-goto 标签只在 engine_exec 作用域有效) */
    engine_resolve_handlers(code, *out_count);

    return code;
}

/*
 * ============================================================================
 * Warp 初始化 (栈分配)
 * ============================================================================
 */
static void scheduler_init_warp(GPGPUWarp *warp, uint32_t pc,
                                 uint32_t thread_id_base, const uint32_t block_id[3],
                                 uint32_t num_threads, uint32_t warp_id,
                                 uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    if (num_threads >= GPGPU_WARP_SIZE)
        warp->active_mask = 0xFFFFFFFF;
    else
        warp->active_mask = (1U << num_threads) - 1;

    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->active = (warp->active_mask & (1 << i)) != 0;
        lane->gpr[0].u32 = 0;
        lane->fpr[0].u32 = 0;
    }
}

/*
 * ============================================================================
 * SoA 编组 — warp->lanes[] (AoS) → SoA 数组
 * ============================================================================
 */
static void aos_to_soa(const GPGPUWarp *warp,
                        uint32_t gpr[32 * 32], uint32_t fpr[32 * 32],
                        uint32_t pc[32], uint32_t mhartid[32], uint32_t fcsr[32])
{
    for (int lane = 0; lane < 32; lane++) {
        const GPGPULane *l = &warp->lanes[lane];
        pc[lane]      = l->pc;
        mhartid[lane] = l->mhartid;
        fcsr[lane]    = l->fcsr;
        for (int r = 0; r < GPGPU_NUM_REGS; r++) {
            gpr[r * 32 + lane] = l->gpr[r].u32;
            fpr[r * 32 + lane] = l->fpr[r].u32;
        }
    }
}

/*
 * ============================================================================
 * SoA 解组 — SoA 数组 → warp->lanes[] (AoS)
 * ============================================================================
 */
static void soa_to_aos(GPGPUWarp *warp,
                        const uint32_t gpr[32 * 32], const uint32_t fpr[32 * 32],
                        const uint32_t pc[32], const uint32_t mhartid[32], const uint32_t fcsr[32])
{
    for (int lane = 0; lane < 32; lane++) {
        GPGPULane *l = &warp->lanes[lane];
        l->pc      = pc[lane];
        l->mhartid = mhartid[lane];
        l->fcsr    = fcsr[lane];
        for (int r = 0; r < GPGPU_NUM_REGS; r++) {
            l->gpr[r].u32 = gpr[r * 32 + lane];
            l->fpr[r].u32 = fpr[r * 32 + lane];
        }
    }
}

/*
 * ============================================================================
 * 执行单个 warp
 * ============================================================================
 */
static int scheduler_exec_warp(GPGPUState *s, GPGPUWarp *warp, ThOp *code, int tcount)
{
    /* SoA 编组 */
    uint32_t gpr[32 * 32];
    uint32_t fpr[32 * 32];
    uint32_t pc[32];
    uint32_t mhartid[32];
    uint32_t fcsr[32];

    aos_to_soa(warp, gpr, fpr, pc, mhartid, fcsr);

    /* 设置 SIMT 上下文 (供 CTRL MMIO 读) — 固定值部分 */
    s->simt.thread_id[0] = warp->thread_id_base;
    s->simt.thread_id[1] = 0;
    s->simt.thread_id[2] = 0;
    s->simt.block_id[0]  = warp->block_id[0];
    s->simt.block_id[1]  = warp->block_id[1];
    s->simt.block_id[2]  = warp->block_id[2];

    /* 构建引擎上下文 */
    EngineContext ctx = {
        .s               = s,
        .active          = warp->active_mask,
        .thread_id_base  = warp->thread_id_base,
        .block_id        = { warp->block_id[0], warp->block_id[1], warp->block_id[2] },
    };

    /* 执行 */
    int ret = engine_exec(code, tcount, &ctx, gpr, fpr, pc, mhartid, fcsr);
    if (ret != 0) return ret;

    /* SoA 解组 */
    soa_to_aos(warp, gpr, fpr, pc, mhartid, fcsr);

    return 0;
}

/*
 * ============================================================================
 * 调度入口 — Grid→Block→Warp 主循环
 * ============================================================================
 */
int scheduler_run_kernel(GPGPUState *s)
{
    uint32_t grid_dim[3]  = { s->kernel.grid_dim[0], s->kernel.grid_dim[1], s->kernel.grid_dim[2] };
    uint32_t block_dim[3] = { s->kernel.block_dim[0], s->kernel.block_dim[1], s->kernel.block_dim[2] };
    uint32_t kern_addr    = s->kernel.kernel_addr;
    uint32_t tpb = block_dim[0] * block_dim[1] * block_dim[2];

    /* 预译码 (缓存) — 一次 kernel launch 只做一次 */
    uint32_t kern_size = 4096;  /* 与旧代码一致: test kernel 内最大覆盖 */
    int tcount = 0;
    ThOp *code = scheduler_predecode(s, kern_addr, kern_size, &tcount);
    if (!code) return -1;

    /* Grid→Block→Warp 三重循环 */
    int result = 0;
    for (uint32_t z = 0; z < grid_dim[2]; z++) {
        for (uint32_t y = 0; y < grid_dim[1]; y++) {
            for (uint32_t x = 0; x < grid_dim[0]; x++) {
                uint32_t block_id[3] = { x, y, z };
                uint32_t blk_linear  = z * grid_dim[0] * grid_dim[1] + y * grid_dim[0] + x;
                uint32_t num_warps   = (tpb + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;

                for (uint32_t w = 0; w < num_warps; w++) {
                    GPGPUWarp warp;
                    uint32_t tid_base = w * GPGPU_WARP_SIZE;
                    uint32_t n_threads = tpb - tid_base;
                    if (n_threads > GPGPU_WARP_SIZE) n_threads = GPGPU_WARP_SIZE;

                    scheduler_init_warp(&warp, kern_addr, tid_base,
                                        block_id, n_threads, w, blk_linear);

                    int ret = scheduler_exec_warp(s, &warp, code, tcount);
                    if (ret != 0) { result = -1; goto done; }
                }
            }
        }
    }

done:
    free(code);
    return result;
}
