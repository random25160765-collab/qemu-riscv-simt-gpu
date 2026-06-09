/*
 * engine_mod.c — RISC-V SIMT 模块化执行引擎
 *
 * 与原 engine.c 的区别:
 *   所有 op_xxx handler 移到 inst/ 层 (ISA 扩展 → 代码生成)
 *   由 module/ 层按硬件模块聚合 (ctrl/alu/fpu/lsu/sfu/vpu/tcu)
 *   engine_mod.c 只保留: 函数框架 + 共享上下文 + dispatch 初始化 + op_illegal
 *
 * 架构:
 *   inst/{isa}/spec.yaml  →  gen_{isa}.py  →  inst/{isa}/handles/ (生成的 .h)
 *   module/{unit}.h       →  聚合 ISA handler → #include 到此处
 *   engine_mod.c          →  computed-goto 解释器 + dispatch 表
 *
 * SoA 数据布局，active 掩码控制 lane 宽度。
 * computed-goto 线程化解释器 + dispatch 表 constructor 一次性填充。
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#include <math.h>
#include "state.h"
#include "gpgpu_core.h"
#include "../inst/dispatch_list.h" /* NEW: INSTRUCTION_LIST + NUM_OF_INST=136 + DISP_ enum */
#include "engine_types.h"          /* SIMTFrame + EngineContext, 无 dispatch 依赖 */
#include "predecode.h"             /* ThOp (其依赖的 dispatch.h 由 #ifndef 守卫跳过) */
#include "lpfp.h"
#include "vram.h"
#include "sfu.h" /* fast math approx */

/* ============================================================
 * SoA 数据布局宏
 * ============================================================ */
#define GPR(reg, lane) gpr[(reg) * 32 + (lane)]
#define FPR(reg, lane) fpr[(reg) * 32 + (lane)]
#define PC(lane) _pc[lane]
#define FOR_EACH_LANE                  \
    for (int _li = 0; _li < 32; _li++) \
        if ((_active >> _li) & 1)

/* float 视图 (SoA)，供 -O3 自动向量化 */
#define FR(reg, lane) ((float *)fpr)[(reg) * 32 + (lane)]
/* 向量寄存器视图 (SoA) */
#define VR(reg, lane) ((float *)vpr)[(reg) * 32 + (lane)]

/* 分支提示 */
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* ============================================================
 * FP 精确异常: 0=关闭(向量化快路径), 1=开启(frm/fflags 完整支持)
 *   开启后 FOR_EACH_LANE 内部有分支, 编译器无法自动向量化 → 性能损失 ~5-8x
 * ============================================================ */
#define ENABLE_FP_CSR 1

#if ENABLE_FP_CSR

/* RISC-V frm → C 舍入函数 */
static inline float fpu_round(float val, int frm)
{
    switch (frm) {
    case 1: return truncf(val); /* RTZ */
    case 2: return floorf(val); /* RDN */
    case 3: return ceilf(val);  /* RUP */
    case 4: return roundf(val); /* RMM */
    default: return val;
    }
}

#define FCSR_FRM(f) (((f) >> 5) & 7)
#define FCSR_FFLAGS(f) ((f) & 0x1F)

static inline void fpu_check_fflags(uint32_t *fcsr, float res, float a, float b, int div_op)
{
    int fl = 0x01;
    if (isnan(res)) {
        if (!(isnan(a) || isnan(b))) fl |= 0x10;
    }
    if (div_op && b == 0.0f && !isnan(a) && !isinf(a)) fl |= 0x08;
    if (isinf(res) && isfinite(a) && isfinite(b) && b != 0.0f) fl |= 0x04;
    if (res == 0.0f && ((a != 0.0f && !isinf(a)) || (b != 0.0f && !isinf(b)))) fl |= 0x02;
    *fcsr |= fl;
}

static inline void fpu_check_fflags1(uint32_t *fcsr, float res, float a)
{
    int fl = 0x01;
    if (isnan(res) && !isnan(a)) fl |= 0x10;
    if (isinf(res) && isfinite(a)) fl |= 0x04;
    *fcsr |= fl;
}

/* CSR 感知的 FP 操作宏 */
#define FP_BIN_CSR(rd, rs1, rs2, op, is_div)                                               \
    do {                                                                                   \
        float _a = FR(rs1, _li), _b = FR(rs2, _li);                                        \
        float _r = _a op _b;                                                               \
        if (UNLIKELY(FCSR_FRM(_fcsr[_li]) != 0)) _r = fpu_round(_r, FCSR_FRM(_fcsr[_li])); \
        if (UNLIKELY(isfinite(_r) == 0 || _b == 0.0f))                                     \
            fpu_check_fflags(&_fcsr[_li], _r, _a, _b, is_div);                             \
        else                                                                               \
            _fcsr[_li] |= 0x01;                                                            \
        FR(rd, _li) = _r;                                                                  \
    } while (0)

#define FP_UNA_CSR(rd, rs1, expr)                                                          \
    do {                                                                                   \
        float _a = FR(rs1, _li);                                                           \
        float _r = (expr);                                                                 \
        if (UNLIKELY(FCSR_FRM(_fcsr[_li]) != 0)) _r = fpu_round(_r, FCSR_FRM(_fcsr[_li])); \
        if (UNLIKELY(isfinite(_r) == 0))                                                   \
            fpu_check_fflags1(&_fcsr[_li], _r, _a);                                        \
        else                                                                               \
            _fcsr[_li] |= 0x01;                                                            \
        FR(rd, _li) = _r;                                                                  \
    } while (0)

#endif /* ENABLE_FP_CSR */

/* ============================================================
 * CTRL 寄存器 per-lane 读取
 * ============================================================ */
static inline uint32_t ctrl_read(const EngineContext *ctx, uint32_t addr, int lane)
{
    GPGPUState *s = ctx->s;
    if (addr < 0x80000000) return gpu_read(s, addr, 4);

    /* shared memory: 0x80001000+ */
    if (addr >= 0x80001000 && ctx->shm) {
        uint32_t off = addr - 0x80001000;
        if (off + 4 <= ctx->shm_size) return *(uint32_t *)(ctx->shm + off);
        return 0;
    }

    switch (addr - 0x80000000) {
    case 0x00: return ctx->thread_id[0] + lane; /* thread_id.x */
    case 0x04: return ctx->thread_id[1];        /* thread_id.y */
    case 0x08: return ctx->thread_id[2];        /* thread_id.z */
    case 0x10: return ctx->block_id[0];
    case 0x14: return ctx->block_id[1];
    case 0x18: return ctx->block_id[2];
    case 0x20: return s->kernel.block_dim[0];
    case 0x24: return s->kernel.block_dim[1];
    case 0x28: return s->kernel.block_dim[2];
    case 0x30: return s->kernel.grid_dim[0];
    case 0x34: return s->kernel.grid_dim[1];
    case 0x38: return s->kernel.grid_dim[2];
    /* perf counters (read-only) */
    case 0x200: return (uint32_t)s->stats.total_warps;
    case 0x204: return (uint32_t)s->stats.kernel_ops;
    case 0x208: return (uint32_t)s->stats.bytes_read;
    case 0x20C: return (uint32_t)s->stats.bytes_write;
    case 0x210: return (uint32_t)s->stats.total_branches;
    case 0x214: return (uint32_t)s->stats.simt_diverges;
    case 0x218: return (uint32_t)s->stats.cat[0]; /* ALU */
    case 0x21C: return (uint32_t)s->stats.cat[1]; /* FP */
    case 0x220: return (uint32_t)s->stats.cat[2]; /* MEM */
    case 0x224: return (uint32_t)s->stats.cat[3]; /* BR */
    default: return 0;
    }
}

/* ============================================================
 * Dispatch 表 — 文件作用域 static, 所有 warp 共享
 *
 * NUM_OF_INST 由 dispatch_list.h (gen_dispatch.py 生成) 提供
 * DISP_ enum 值与 INSTRUCTION_LIST 顺序对齐
 * ============================================================ */
static void *dispatch[NUM_OF_INST];
static volatile int dispatch_ready = 0;
/* perf_enabled: 首次 engine_exec 时从 s->cfg.features.perf 读取。
 * static 变量, 程序生命周期内不变。
 * Makefile 中 engine.o 依赖 gpu_config.lua — 改 perf 后 make 自动重编。 */
static int perf_enabled = -1;

/* 热路径计数器包装: 分支预测 100% 命中 ≈ 零开销 */
#define PERF_IF(x)      \
    if (perf_enabled) { \
        x;              \
    }

/* 合并分析: warp 内 32 lane 访存是否在同 64B cache line */
#define COALESCE(rs1, imm)                               \
    do {                                                 \
        if (perf_enabled) {                              \
            uint32_t _mn = 0xFFFFFFFF, _mx = 0;          \
            for (int _l = 0; _l < 32; _l++)              \
                if ((_active >> _l) & 1) {               \
                    uint32_t _ad = GPR(rs1, _l) + (imm); \
                    if (_ad < _mn) _mn = _ad;            \
                    if (_ad > _mx) _mx = _ad;            \
                }                                        \
            if ((_mx - _mn) < 64) s->stats.coal_ops++;   \
            s->stats.coal_total++;                       \
        }                                                \
    } while (0)
void engine_resolve_handlers(ThOp *code, int tcount)
{
    (void)code;
    (void)tcount;
    /* 实际解析在 engine_exec 内部进行 (lazy init + resolve) */
}

/* ============================================================
 * 引擎入口
 * ============================================================ */
int engine_exec(ThOp *code, int tcount, const EngineContext *ctx, uint32_t gpr[32 * 32], uint32_t fpr[32 * 32],
                uint32_t *vpr, uint32_t *vl, uint32_t _pc[32], uint32_t _mhartid[32], uint32_t _fcsr[32],
                SIMTFrame *_stk_ext, int *_sdepth_ext, int resume_pc, float _mma_acc[8 * 32])
{
    ThOp *ip;
    uint32_t _active = ctx->active;
    uint32_t _vl = vl ? *vl : 32;
    GPGPUState *s = ctx->s;
    (void)tcount;

/* 短别名供 CSR/LP 使用 */
#define MHARTID(l) _mhartid[l]
#define FCSR(l) _fcsr[l]

/* 从外部指针初始化 SIMT stack 局部变量（热路径零间接访存） */
#define SIMT_STACK_MAX 32
    struct {
        int32_t ft_idx;
        uint32_t mask;
    } _stk[SIMT_STACK_MAX];
    int _sdepth;
    if (resume_pc < 0) {
        _sdepth = 0;
    } else {
        memcpy(_stk, _stk_ext, sizeof(_stk));
        _sdepth = *_sdepth_ext;
    }

    /* === lazy init: 填充 dispatch 表 (DISP_ enum 索引) === */
    if (!dispatch_ready) {
        int di = 0;
        /* 标量指令: 顺序填充 dispatch[], 与 DISP_ enum 值天然对齐 */
#define X(name, pattern, op_type, imm_fn) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
#undef X

        /* 向量指令: rvv_ 前缀, 默认 SEW=32; vsetvli 时 rebind */
#include "../inst/dispatch_rebind.h"
        if (s->cfg.features.vpu) {
            RVV_DISPATCH_REBIND(dispatch, 32);
        }

        /* perf flag: 一次性从 config 读 */
        if (perf_enabled == -1) perf_enabled = s->cfg.features.perf ? 1 : 0;

        dispatch_ready = 1;
    }

    /* resume from barrier: 跳过 handler 解析, 直接跳到保存的 ip */
    if (resume_pc >= 0) {
        ip = &code[resume_pc];
    } else {
        if (code[0].handler == NULL || (uintptr_t)code[0].handler < 256) {
            for (int i = 0; i <= tcount; i++) {
                uint16_t id = (uint16_t)(uintptr_t)code[i].handler;
                if (id > 0 && id < NUM_OF_INST)
                    code[i].handler = dispatch[id - 1];
                else if (i < tcount)
                    code[i].handler = &&op_illegal;
                else
                    code[i].handler = &&op_done;
            }
        }
        ip = code;
    }

/* x0 清零 */
#define NEXT()                                                                         \
    do {                                                                               \
        gpr[0 * 32 + 0] = gpr[0 * 32 + 1] = gpr[0 * 32 + 2] = gpr[0 * 32 + 3] = 0;     \
        gpr[0 * 32 + 4] = gpr[0 * 32 + 5] = gpr[0 * 32 + 6] = gpr[0 * 32 + 7] = 0;     \
        gpr[0 * 32 + 8] = gpr[0 * 32 + 9] = gpr[0 * 32 + 10] = gpr[0 * 32 + 11] = 0;   \
        gpr[0 * 32 + 12] = gpr[0 * 32 + 13] = gpr[0 * 32 + 14] = gpr[0 * 32 + 15] = 0; \
        gpr[0 * 32 + 16] = gpr[0 * 32 + 17] = gpr[0 * 32 + 18] = gpr[0 * 32 + 19] = 0; \
        gpr[0 * 32 + 20] = gpr[0 * 32 + 21] = gpr[0 * 32 + 22] = gpr[0 * 32 + 23] = 0; \
        gpr[0 * 32 + 24] = gpr[0 * 32 + 25] = gpr[0 * 32 + 26] = gpr[0 * 32 + 27] = 0; \
        gpr[0 * 32 + 28] = gpr[0 * 32 + 29] = gpr[0 * 32 + 30] = gpr[0 * 32 + 31] = 0; \
        PERF_IF(s->stats.cat[ip[-1].cat]++;);                                          \
        goto *ip++->handler;                                                           \
    } while (0)

    goto *ip++->handler; /* 首条: 绕开 NEXT() 的 ip[-1] 越界 */

/*
	 * ============================================================
	 * 模块化 handler 聚合入口
	 * ============================================================ */
#include "../module/modules.h"

/* ============================================================
	 * 错误出口
	 * ============================================================ */
op_illegal: {
    return -1;
}
}
