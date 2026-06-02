/*
 * scheduler.c — GPU Kernel Scheduler Implementation
 *
 * Copyright (c) 2024-2025  Licensed under GPL v2 or later.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "state.h"
#include "gpgpu_core.h"
#include "engine.h"
#include "scheduler.h"
#include "predecode.h"
#include "dispatch.h"
#include "vram.h"
#include "soa.h"

#define SIMT_STACK_MAX 32

/* ============================================================
 * WarpSlot — barrier 持久化状态
 * ============================================================ */
typedef struct {
    uint32_t gpr[32 * 32], fpr[32 * 32], vpr[32 * 32], pc[32], mhartid[32], fcsr[32];
    float mma_acc[8 * 32];
    SIMTFrame stk[SIMT_STACK_MAX];
    int sdepth;
    uint32_t active;
    int resume_pc;
    int ret;
    bool done;   /* warp 已执行完毕 */
    bool at_bar; /* warp 停在 barrier */
} WarpSlot;

/* ============================================================
 * BlockContext
 * ============================================================ */
typedef struct BlockContext {
    GPGPUState *s;
    ThOp *code;
    int tcount;
    uint32_t kern_addr, tpb;
    uint32_t block_id[3], blk_linear;
    uint32_t num_warps;
    uint8_t *shm;
    uint32_t shm_size;
    bool has_vpr; /* kernel contains RVV instructions (opcode 0x57) */
    WarpSlot *slots;
    int ret;
} BlockContext;

static ThOp *scheduler_predecode(GPGPUState *s, uint32_t kern_addr, uint32_t kern_size, int *out_count)
{
    SIMDDecoder dec = {0};
    simd_decoder_init(&dec);
    ThOp *code = simd_predecode(&dec, s, kern_addr, kern_size, out_count);
    if (!code) return NULL;
    engine_resolve_handlers(code, *out_count);
    return code;
}

static void scheduler_init_warp(GPGPUWarp *warp, uint32_t pc, uint32_t tid_base, const uint32_t bid[3], uint32_t n_thr,
                                uint32_t wid, uint32_t blk_lin)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = tid_base;
    warp->warp_id = wid;
    warp->block_id[0] = bid[0];
    warp->block_id[1] = bid[1];
    warp->block_id[2] = bid[2];
    warp->active_mask = (n_thr >= GPGPU_WARP_SIZE) ? 0xFFFFFFFF : (1U << n_thr) - 1;
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *l = &warp->lanes[i];
        l->pc = pc;
        l->mhartid = MHARTID_ENCODE(blk_lin, wid, i);
        l->active = (warp->active_mask & (1 << i)) != 0;
        l->gpr[0].u32 = 0;
        l->fpr[0].u32 = 0;
    }
}

/* ---- 从 slot 恢复 warp 状态到 SoA ---- */
static void slot_to_soa(WarpSlot *slot, uint32_t *gpr, uint32_t *fpr, uint32_t *vpr, uint32_t *pc, uint32_t *mh,
                        uint32_t *fcsr, SIMTFrame *stk, int *sdepth, uint32_t *active, int *rpc, float *mma)
{
    memcpy(gpr, slot->gpr, 32 * 32 * 4);
    memcpy(fpr, slot->fpr, 32 * 32 * 4);
    if (vpr) memcpy(vpr, slot->vpr, 32 * 32 * 4);
    memcpy(pc, slot->pc, 32 * 4);
    memcpy(mh, slot->mhartid, 32 * 4);
    memcpy(fcsr, slot->fcsr, 32 * 4);
    memcpy(mma, slot->mma_acc, 8 * 32 * 4);
    memcpy(stk, slot->stk, SIMT_STACK_MAX * 8);
    *sdepth = slot->sdepth;
    *active = slot->active;
    *rpc = slot->resume_pc;
}

/* ---- 保存 SoA 到 slot ---- */
static void soa_to_slot(WarpSlot *slot, uint32_t *gpr, uint32_t *fpr, uint32_t *vpr, uint32_t *pc, uint32_t *mh,
                        uint32_t *fcsr, SIMTFrame *stk, int sdepth, uint32_t active, int rpc, float *mma)
{
    memcpy(slot->gpr, gpr, 32 * 32 * 4);
    memcpy(slot->fpr, fpr, 32 * 32 * 4);
    if (vpr) memcpy(slot->vpr, vpr, 32 * 32 * 4);
    memcpy(slot->pc, pc, 32 * 4);
    memcpy(slot->mhartid, mh, 32 * 4);
    memcpy(slot->fcsr, fcsr, 32 * 4);
    memcpy(slot->mma_acc, mma, 8 * 32 * 4);
    memcpy(slot->stk, stk, SIMT_STACK_MAX * 8);
    slot->sdepth = sdepth;
    slot->active = active;
    slot->resume_pc = rpc;
}

/* ============================================================
 * 串行模拟并行: 所有 warp 轮流跑到 barrier → 全到后继续
 * ============================================================ */
static void exec_block(BlockContext *blk)
{
    GPGPUState *s = blk->s;
    blk->slots = calloc(blk->num_warps, sizeof(WarpSlot));
    blk->ret = 0;

    /* 为每个 warp 初始化独立的 warp 对象和 SoA 状态 */
    GPGPUWarp *warps = calloc(blk->num_warps, sizeof(GPGPUWarp));
    uint32_t(*gpr)[32 * 32] = calloc(blk->num_warps, 32 * 32 * 4);
    uint32_t(*fpr)[32 * 32] = calloc(blk->num_warps, 32 * 32 * 4);
    uint32_t(*vpr)[32 * 32] = blk->has_vpr ? calloc(blk->num_warps, 32 * 32 * 4) : NULL;
    uint32_t(*pc)[32] = calloc(blk->num_warps, 32 * 4);
    uint32_t(*mh)[32] = calloc(blk->num_warps, 32 * 4);
    uint32_t(*fcsr)[32] = calloc(blk->num_warps, 32 * 4);
    SIMTFrame(*stk)[32] = calloc(blk->num_warps, 32 * sizeof(SIMTFrame));
    int *sdepth = calloc(blk->num_warps, sizeof(int));
    float(*mma)[8 * 32] = calloc(blk->num_warps, 8 * 32 * 4);
    EngineContext *ctxs = calloc(blk->num_warps, sizeof(EngineContext));
    int *resume_pc = calloc(blk->num_warps, sizeof(int));

    /* 初始化所有 warp */
    for (uint32_t w = 0; w < blk->num_warps; w++) {
        uint32_t tb = w * GPGPU_WARP_SIZE, nt = blk->tpb - tb;
        if (nt > GPGPU_WARP_SIZE) nt = GPGPU_WARP_SIZE;
        scheduler_init_warp(&warps[w], blk->kern_addr, tb, blk->block_id, nt, w, blk->blk_linear);
        aos_to_soa(&warps[w], gpr[w], fpr[w], vpr ? vpr[w] : NULL, pc[w], mh[w], fcsr[w]);
        resume_pc[w] = -1;
        ctxs[w] = (EngineContext){
                .s = s,
                .active = warps[w].active_mask,
                .shm = blk->shm,
                .shm_size = blk->shm_size,
                .thread_id = {warps[w].thread_id_base, 0, 0},
                .block_id = {warps[w].block_id[0], warps[w].block_id[1], warps[w].block_id[2]},
                .warp_id = warps[w].warp_id,
                .thread_mask = warps[w].active_mask,
        };
    }

    /* 主循环: 每次迭代推进所有 warp 到下一个 barrier 或完成 */
    for (;;) {
        int n_done = 0, n_at_bar = 0;

        /* Round 1: 每个未完成的 warp 执行到 barrier 或 done */
        for (uint32_t w = 0; w < blk->num_warps; w++) {
            if (blk->slots[w].done) {
                n_done++;
                continue;
            }

            int ret = engine_exec(blk->code, blk->tcount, &ctxs[w], gpr[w], fpr[w], vpr ? vpr[w] : NULL, pc[w], mh[w],
                                  fcsr[w], stk[w], &sdepth[w], resume_pc[w], mma[w]);

            if (ret & 0x10000) {
                /* 到达 barrier: 保存状态 */
                soa_to_slot(&blk->slots[w], gpr[w], fpr[w], vpr ? vpr[w] : NULL, pc[w], mh[w], fcsr[w], stk[w],
                            sdepth[w], ctxs[w].active, ret & 0xFFFF, mma[w]);
                blk->slots[w].at_bar = true;
                blk->slots[w].done = false;
                n_at_bar++;
            } else {
                /* done 或 error */
                blk->slots[w].ret = ret;
                blk->slots[w].done = true;
                blk->slots[w].at_bar = false;
                if (ret != 0) blk->ret = -1;
                n_done++;
            }
        }

        if (n_done == (int)blk->num_warps) break; /* 全部完成 */

        /* Round 2: 所有在 barrier 的 warp 恢复并继续 */
        if (n_at_bar > 0) {
            for (uint32_t w = 0; w < blk->num_warps; w++) {
                if (!blk->slots[w].at_bar) continue;
                blk->slots[w].at_bar = false;
                slot_to_soa(&blk->slots[w], gpr[w], fpr[w], vpr ? vpr[w] : NULL, pc[w], mh[w], fcsr[w], stk[w],
                            &sdepth[w], &ctxs[w].active, &resume_pc[w], mma[w]);
            }
        }
    }

    /* 写回结果 */
    for (uint32_t w = 0; w < blk->num_warps; w++) {
        soa_to_aos(&warps[w], gpr[w], fpr[w], vpr ? vpr[w] : NULL, pc[w], mh[w], fcsr[w]);
        blk->slots[w].ret = blk->slots[w].done ? 0 : -1;
    }

    free(warps);
    free(gpr);
    free(fpr);
    free(vpr);
    free(pc);
    free(mh);
    free(fcsr);
    free(stk);
    free(sdepth);
    free(mma);
    free(ctxs);
    free(resume_pc);
    free(blk->slots);
    blk->slots = NULL;
}

/* ============================================================
 * Block 级线程池
 * ============================================================ */
typedef struct {
    volatile int *idx;
    BlockContext *blocks;
    int total;
} WorkQueue;

static void *pool_worker(void *arg)
{
    WorkQueue *wq = (WorkQueue *)arg;
    while (1) {
        int i = __sync_fetch_and_add(wq->idx, 1);
        if (i >= wq->total) break;
        exec_block(&wq->blocks[i]);
    }
    return NULL;
}

/* ============================================================
 * 调度入口
 * ============================================================ */
int scheduler_run_kernel(GPGPUState *s)
{
    memset(&s->stats, 0, sizeof(s->stats));
    cache_reset();

    uint32_t gd[3] = {s->kernel.grid_dim[0], s->kernel.grid_dim[1], s->kernel.grid_dim[2]};
    uint32_t bd[3] = {s->kernel.block_dim[0], s->kernel.block_dim[1], s->kernel.block_dim[2]};
    uint32_t tpb = bd[0] * bd[1] * bd[2];

    int tcount = 0;
    ThOp *code = scheduler_predecode(s, s->kernel.kernel_addr, s->kern_size ? s->kern_size : 4096, &tcount);
    if (!code) return -1;

    bool has_vpr = false;
    for (int i = 0; i < tcount; i++) {
        uint32_t inst = code[i].inst;
        int c = CAT_ALU;
        if (inst == 0)
            c = CAT_SYS;
        else
            switch (inst & 0x7F) {
            case 0x03: c = CAT_MEM; break;
            case 0x07:
                c = CAT_MEM;
                if (((inst >> 12) & 7) == 6) has_vpr = true; /* vle32.v */
                break;
            case 0x23: c = CAT_MEM; break;
            case 0x27:
                c = CAT_MEM;
                if (((inst >> 12) & 7) == 6) has_vpr = true; /* vse32.v */
                break;
            case 0x63:
            case 0x67:
            case 0x6F: c = CAT_BR; break;
            case 0x43:
            case 0x47:
            case 0x4B:
            case 0x4F:
            case 0x53: c = CAT_FP; break;
            case 0x73: c = CAT_SYS; break;
            case 0x2B: c = (((inst >> 12) & 7) == 7) ? CAT_TCU : CAT_VPU; break;
            case 0x57:
                c = CAT_VPU;
                has_vpr = true;
                break; /* RVV standard */
            }
        code[i].cat = (uint8_t)c;
    }
    memset(s->stats.cat, 0, sizeof(s->stats.cat));
    memset(s->stats.cat_static, 0, sizeof(s->stats.cat_static));
    for (int i = 0; i < tcount; i++)
        s->stats.cat_static[code[i].cat]++;
    s->stats.kernel_ops = (uint64_t)tcount;

    uint32_t total_blocks = gd[0] * gd[1] * gd[2];
    BlockContext *blocks = calloc(total_blocks, sizeof(BlockContext));
    if (!blocks) {
        free(code);
        return -1;
    }
    int nb = 0;
    for (uint32_t z = 0; z < gd[2]; z++)
        for (uint32_t y = 0; y < gd[1]; y++)
            for (uint32_t x = 0; x < gd[0]; x++) {
                BlockContext *b = &blocks[nb++];
                b->s = s;
                b->code = code;
                b->tcount = tcount;
                b->kern_addr = s->kernel.kernel_addr;
                b->tpb = tpb;
                b->block_id[0] = x;
                b->block_id[1] = y;
                b->block_id[2] = z;
                b->blk_linear = z * gd[0] * gd[1] + y * gd[0] + x;
                b->num_warps = (tpb + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;
                b->has_vpr = has_vpr;
                s->stats.total_warps += b->num_warps;
                b->shm_size = s->kernel.shared_mem_size;
                if (b->shm_size > 0) b->shm = calloc(1, b->shm_size);
            }

    if (nb > 1) {
        int nw = s->cfg.num_cus ? (int)(s->cfg.num_cus * s->cfg.warps_per_cu) : 0;
        if (!nw) nw = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nw < 1) nw = 4;
        if (nw > nb) nw = nb;
        int idx = 0;
        WorkQueue wq = {&idx, blocks, nb};
        pthread_t *th = calloc(nw, sizeof(pthread_t));
        for (int i = 0; i < nw; i++)
            pthread_create(&th[i], NULL, pool_worker, &wq);
        for (int i = 0; i < nw; i++)
            pthread_join(th[i], NULL);
        free(th);
    } else
        exec_block(&blocks[0]);

    int result = 0;
    for (int i = 0; i < nb; i++) {
        if (blocks[i].ret != 0) result = -1;
        free(blocks[i].shm);
    }
    free(blocks);
    free(code);
    return result;
}
