/*
 * scheduler.c — GPU Kernel Scheduler Implementation
 *
 * 职责:
 *   1. 预译码 — kernel binary → ThOp[]
 *   2. Grid→Block→Warp dispatch (pthread 并行 block)
 *   3. Warp 状态初始化 + SoA 编组/解组
 *   4. SIMT 上下文注入 (线程安全, 存在 EngineContext)
 *   5. Shared memory 分配 + barrier 协调
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "state.h"
#include "gpgpu_core.h"
#include "engine.h"
#include "scheduler.h"
#include "simd_predecode.h"
#include "simd_dispatch.h"
#include "memory.h"

/* ============================================================
 * 预译码
 * ============================================================ */
static ThOp *scheduler_predecode(GPGPUState *s, uint32_t kern_addr,
                                  uint32_t kern_size, int *out_count)
{
    SIMDDecoder dec;
    simd_decoder_init(&dec);
    ThOp *code = simd_predecode(&dec, s, kern_addr, kern_size, out_count);
    if (!code) return NULL;
    engine_resolve_handlers(code, *out_count);
    return code;
}

/* ============================================================
 * Warp 初始化
 * ============================================================ */
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

/* ============================================================
 * SoA 编组 / 解组
 * ============================================================ */
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

/* ============================================================
 * Per-block 上下文 (线程安全: 每个 block 独立一份)
 * ============================================================ */
typedef struct {
    GPGPUState *s;
    ThOp       *code;
    int         tcount;
    uint32_t    kern_addr;
    uint32_t    tpb;            /* threads per block */
    uint32_t    block_id[3];
    uint32_t    blk_linear;
    uint32_t    num_warps;

    /* shared memory */
    uint8_t    *shm;
    uint32_t    shm_size;

    /* barrier */
    uint32_t    barrier_target;
    uint32_t    barrier_count;
    bool        barrier_active;

    /* result */
    int         ret;
} BlockContext;

/* ============================================================
 * 执行单个 warp (无 s->simt 访问, 线程安全)
 * ============================================================ */
static int exec_warp(GPGPUState *s, GPGPUWarp *warp, ThOp *code, int tcount,
                      BlockContext *blk)
{
    uint32_t gpr[32 * 32], fpr[32 * 32], pc[32], mhartid[32], fcsr[32];
    aos_to_soa(warp, gpr, fpr, pc, mhartid, fcsr);

    EngineContext ctx = {
        .s = s, .active = warp->active_mask,
        .shm = blk->shm, .shm_size = blk->shm_size,
        .thread_id = { warp->thread_id_base, 0, 0 },
        .block_id  = { warp->block_id[0], warp->block_id[1], warp->block_id[2] },
        .warp_id   = warp->warp_id,
        .thread_mask = warp->active_mask,
    };

    int ret = engine_exec(code, tcount, &ctx, gpr, fpr, pc, mhartid, fcsr);

    if (ret == 1) {
        /* barrier reached */
        blk->barrier_count++;
        if (blk->barrier_count >= blk->barrier_target) {
            blk->barrier_count  = 0;
            blk->barrier_active = false;
        }
        /* FIXME: warp pause/resume needs WarpSlot persistence */
        ret = 0;
    }

    soa_to_aos(warp, gpr, fpr, pc, mhartid, fcsr);
    return ret;
}

/* ============================================================
 * 执行单个 block (pthread 入口, 也可串行调用)
 * ============================================================ */
static void *exec_block(void *arg)
{
    BlockContext *blk = (BlockContext *)arg;
    GPGPUState *s = blk->s;

    for (uint32_t w = 0; w < blk->num_warps; w++) {
        GPGPUWarp warp;
        uint32_t tid_base = w * GPGPU_WARP_SIZE;
        uint32_t n_threads = blk->tpb - tid_base;
        if (n_threads > GPGPU_WARP_SIZE) n_threads = GPGPU_WARP_SIZE;

        scheduler_init_warp(&warp, blk->kern_addr, tid_base,
                            blk->block_id, n_threads, w, blk->blk_linear);

        int ret = exec_warp(s, &warp, blk->code, blk->tcount, blk);
        if (ret != 0) { blk->ret = -1; return NULL; }
    }

    blk->ret = 0;
    return NULL;
}

/* ============================================================
 * 调度入口 — 收集 blocks → pthread 并行
 * ============================================================ */
#define MAX_BLOCKS 512

int scheduler_run_kernel(GPGPUState *s)
{
    uint32_t gd[3] = { s->kernel.grid_dim[0], s->kernel.grid_dim[1], s->kernel.grid_dim[2] };
    uint32_t bd[3] = { s->kernel.block_dim[0], s->kernel.block_dim[1], s->kernel.block_dim[2] };
    uint32_t kern_addr = s->kernel.kernel_addr;
    uint32_t tpb = bd[0] * bd[1] * bd[2];

    int tcount = 0;
    ThOp *code = scheduler_predecode(s, kern_addr, 4096, &tcount);
    if (!code) return -1;

    /* 收集所有 block */
    BlockContext blocks[MAX_BLOCKS];
    int num_blocks = 0;

    for (uint32_t z = 0; z < gd[2]; z++) {
        for (uint32_t y = 0; y < gd[1]; y++) {
            for (uint32_t x = 0; x < gd[0]; x++) {
                if (num_blocks >= MAX_BLOCKS) goto overflow;

                BlockContext *blk = &blocks[num_blocks++];
                memset(blk, 0, sizeof(*blk));
                blk->s         = s;
                blk->code      = code;
                blk->tcount    = tcount;
                blk->kern_addr = kern_addr;
                blk->tpb       = tpb;
                blk->block_id[0] = x; blk->block_id[1] = y; blk->block_id[2] = z;
                blk->blk_linear  = z * gd[0] * gd[1] + y * gd[0] + x;
                blk->num_warps   = (tpb + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;
                blk->barrier_target = blk->num_warps;

                /* shared memory (per-block) */
                blk->shm_size = s->kernel.shared_mem_size;
                if (blk->shm_size > 0)
                    blk->shm = calloc(1, blk->shm_size);
            }
        }
    }

    /* 并行执行所有 block */
    if (num_blocks > 1) {
        pthread_t threads[MAX_BLOCKS];
        for (int i = 0; i < num_blocks; i++)
            pthread_create(&threads[i], NULL, exec_block, &blocks[i]);
        for (int i = 0; i < num_blocks; i++)
            pthread_join(threads[i], NULL);
    } else if (num_blocks == 1) {
        exec_block(&blocks[0]);
    }

    /* 收集结果 + 释放 shm */
    int result = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (blocks[i].ret != 0) result = -1;
        free(blocks[i].shm);
    }

    free(code);
    return result;

overflow:
    free(code);
    return -1;
}
