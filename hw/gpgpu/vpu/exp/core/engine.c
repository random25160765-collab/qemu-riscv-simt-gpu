/*
 * engine.c — RISC-V SIMT 执行引擎 (合并 threaded + simd)
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
#include "engine.h"
#include "lpfp.h"
#include "memory.h"
#include "dispatch.h"  /* INSTRUCTION_LIST (唯一来源) */
#include "predecode.h" /* ThOp */
#include "sfu.h"       /* fast math approx */

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

/* 分支提示 */
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* ============================================================
 * FP 精确异常: 0=关闭(向量化快路径), 1=开启(frm/fflags 完整支持)
 *   开启后 FOR_EACH_LANE 内部有分支, 编译器无法自动向量化 → 性能损失 ~5-8x
 * ============================================================ */
#define ENABLE_FP_CSR 0

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
 * Dispatch 表 + Handler 解析
 * 必须在 engine_exec 内部，因为 &&op_* 标签是函数作用域
 * ============================================================ */
#define NUM_OF_INST 300
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
                uint32_t _pc[32], uint32_t _mhartid[32], uint32_t _fcsr[32], SIMTFrame *_stk_ext, int *_sdepth_ext,
                int resume_pc, float _mma_acc[8 * 32])
{
    ThOp *ip;
    uint32_t _active = ctx->active;
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

    /* === lazy init: 填充 dispatch 表 === */
    if (!dispatch_ready) {
        size_t di = 0;
#define X(name, pattern, op_type, imm_fn) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
#undef X

        /* perf flag: 一次性从 config 读 */
        if (perf_enabled == -1) perf_enabled = s->cfg.features.perf ? 1 : 0;

        /* 特性开关: 直接改 dispatch 表, 热路径零开销 */
        if (!s->cfg.features.sfu) {
            for (int _j = 0; _j < NUM_OF_INST; _j++) {
                void *h = dispatch[_j];
                if (h == &&op_fexp_s || h == &&op_fln_s || h == &&op_frcp_s || h == &&op_frsqrt_s ||
                    h == &&op_ftanh_s || h == &&op_fsigmoid_s || h == &&op_fsin_s || h == &&op_fcos_s)
                    dispatch[_j] = &&op_illegal;
            }
        }
        if (!s->cfg.features.lp) {
            for (int _j = 0; _j < NUM_OF_INST; _j++) {
                void *h = dispatch[_j];
                if (h == &&op_fcvt_s_bf16 || h == &&op_fcvt_bf16_s || h == &&op_fcvt_s_e4m3 || h == &&op_fcvt_e4m3_s ||
                    h == &&op_fcvt_s_e5m2 || h == &&op_fcvt_e5m2_s || h == &&op_fcvt_s_e2m1 || h == &&op_fcvt_e2m1_s)
                    dispatch[_j] = &&op_illegal;
            }
        }
        if (!s->cfg.features.vpu) {
            for (int _j = 0; _j < NUM_OF_INST; _j++) {
                void *h = dispatch[_j];
                if (h == &&op_vld_v || h == &&op_vst_v || h == &&op_vfadd_v || h == &&op_vfsub_v || h == &&op_vfmul_v ||
                    h == &&op_vfdiv_v || h == &&op_vffma_v || h == &&op_vfmul_vs || h == &&op_vfadd_vs ||
                    h == &&op_vfdiv_vs || h == &&op_vfexp_v || h == &&op_vfsig_v || h == &&op_vftanh_v ||
                    h == &&op_vfsqrt_v || h == &&op_vredsum_v || h == &&op_vredmax_v)
                    dispatch[_j] = &&op_illegal;
            }
        }
        if (!s->cfg.features.tcu) {
            for (int _j = 0; _j < NUM_OF_INST; _j++) {
                void *h = dispatch[_j];
                if (h == &&op_mma_cfg || h == &&op_mma_s || h == &&op_mma_zero || h == &&op_mma_ld ||
                    h == &&op_mma_st || h == &&op_mma_relu || h == &&op_mma_bias)
                    dispatch[_j] = &&op_illegal;
            }
        }
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

/* ============================================================
     * 无条件跳转 (所有 lane 统一, 不发散)
     * ============================================================ */
op_jal: {
    int rd = ip[-1].rd, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = PC(_li) + 4;
        PC(_li) += imm;
    }
    ip = &code[ip[-1].branch_tgt];
    NEXT();
}
op_jalr: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    int32_t target = (int32_t)(GPR(rs1, 0) + imm) & ~1;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = PC(_li) + 4;
        PC(_li) = (uint32_t)target;
    }
    ip = &code[(target - s->kernel.kernel_addr) / 4];
    NEXT();
}

/* ============================================================
     * 条件分支 — per-lane 求值 + SIMT stack 分歧处理
     *
     * 对每 lane 求条件 → 分成 taken / not_taken 两组。
     * 双向分歧: push not_taken 到栈 → 先跑 taken → taken 全 ebreak 后 pop 回来跑 not_taken。
     * 单向: 正常跳/顺序。
     *
     * PC 更新: 在改 _active 之前对两组 lane 分别更新 (taken → PC+imm, not_taken → PC+4)。
     * ============================================================ */
#define DIV_BR(cond)                                                        \
    do {                                                                    \
        int _rs1 = ip[-1].rs1, _rs2 = ip[-1].rs2;                           \
        uint32_t _taken = 0;                                                \
        for (int _l = 0; _l < 32; _l++)                                     \
            if ((_active >> _l) & 1) {                                      \
                uint32_t _a = GPR(_rs1, _l), _b = GPR(_rs2, _l);            \
                if (cond) _taken |= (1u << _l);                             \
            }                                                               \
        uint32_t _not_taken = _active & ~_taken;                            \
        /* PC 更新: 必须在改 _active 之前, 对两组分别处理 */ \
        for (int _l = 0; _l < 32; _l++) {                                   \
            if (_taken & (1u << _l))                                        \
                PC(_l) += ip[-1].imm;                                       \
            else if (_not_taken & (1u << _l))                               \
                PC(_l) += 4;                                                \
        }                                                                   \
        if (perf_enabled) s->stats.total_branches++;                        \
        if (_taken && _not_taken) {                                         \
            /* 分歧: push not_taken, 先跑 taken */                      \
            if (_sdepth >= SIMT_STACK_MAX) return -1;                       \
            if (perf_enabled) s->stats.simt_diverges++;                     \
            _stk[_sdepth].ft_idx = (int32_t)(ip - code);                    \
            _stk[_sdepth].mask = _not_taken;                                \
            _sdepth++;                                                      \
            _active = _taken;                                               \
            ip = &code[ip[-1].branch_tgt];                                  \
        } else if (_taken) {                                                \
            _active = _taken;                                               \
            ip = &code[ip[-1].branch_tgt];                                  \
        } else {                                                            \
            _active = _not_taken;                                           \
        }                                                                   \
        NEXT();                                                             \
    } while (0)

op_beq:
    DIV_BR(_a == _b);
op_bne:
    DIV_BR(_a != _b);
op_blt:
    DIV_BR((int32_t)_a < (int32_t)_b);
op_bge:
    DIV_BR((int32_t)_a >= (int32_t)_b);
op_bltu:
    DIV_BR(_a < _b);
op_bgeu:
    DIV_BR(_a >= _b);
#undef DIV_BR

/* ============================================================
     * Load — per-lane 地址，per-lane CTRL 读
     * ============================================================ */
op_lb: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        uint8_t v = (uint8_t)gpu_read(s, a, 1);
        GPR(rd, _li) = (uint32_t)((int32_t)(v << 24) >> 24);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 1;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}
op_lh: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        uint16_t v = (uint16_t)gpu_read(s, a, 2);
        GPR(rd, _li) = (uint32_t)((int32_t)(v << 16) >> 16);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 2;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}
op_lw: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        GPR(rd, _li) = ctrl_read(ctx, a, _li);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 4;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}
op_lbu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        GPR(rd, _li) = (uint32_t)gpu_read(s, a, 1);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 1;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}
op_lhu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        GPR(rd, _li) = (uint32_t)gpu_read(s, a, 2);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 2;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}

/* Store */
op_sb: {
    int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        gpu_write(s, GPR(rs1, _li) + imm, 1, GPR(rs2, _li));
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 1;)
    NEXT();
}
op_sh: {
    int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        gpu_write(s, GPR(rs1, _li) + imm, 2, GPR(rs2, _li));
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 2;)
    NEXT();
}
op_sw: {
    int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        gpu_write(s, GPR(rs1, _li) + imm, 4, GPR(rs2, _li));
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 4;)
    NEXT();
}

/* ============================================================
     * Upper immediate
     * ============================================================ */
op_lui: {
    int rd = ip[-1].rd;
    uint32_t v = (uint32_t)ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = v;
        PC(_li) += 4;
    }
    NEXT();
}
op_auipc: {
    int rd = ip[-1].rd;
    uint32_t v = (uint32_t)ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = PC(_li) + v;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * ALU immediate
     * ============================================================ */
#define ALUI(name, op)                                          \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm; \
        FOR_EACH_LANE                                           \
        {                                                       \
            GPR(rd, _li) = GPR(rs1, _li) op(uint32_t) imm;      \
            PC(_li) += 4;                                       \
        }                                                       \
        NEXT();                                                 \
    }
    ALUI(addi, +)
    ALUI(xori, ^)
    ALUI(ori, |)
    ALUI(andi, &)
#undef ALUI

op_slti: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = ((int32_t)GPR(rs1, _li) < imm) ? 1 : 0;
        PC(_li) += 4;
    }
    NEXT();
}
op_sltiu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (GPR(rs1, _li) < (uint32_t)imm) ? 1 : 0;
        PC(_li) += 4;
    }
    NEXT();
}
op_slli: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = GPR(rs1, _li) << (imm & 0x1F);
        PC(_li) += 4;
    }
    NEXT();
}
op_srli: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = GPR(rs1, _li) >> (imm & 0x1F);
        PC(_li) += 4;
    }
    NEXT();
}
op_srai: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (uint32_t)((int32_t)GPR(rs1, _li) >> (imm & 0x1F));
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * ALU register
     * ============================================================ */
#define ALUR(name, op)                                          \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            GPR(rd, _li) = GPR(rs1, _li) op GPR(rs2, _li);      \
            PC(_li) += 4;                                       \
        }                                                       \
        NEXT();                                                 \
    }
    ALUR(add, +)
    ALUR(sub, -)
    ALUR(xor, ^) ALUR(or, |) ALUR(and, &) ALUR(mul, *)
#undef ALUR

            op_sll:
    {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {
            GPR(rd, _li) = GPR(rs1, _li) << (GPR(rs2, _li) & 0x1F);
            PC(_li) += 4;
        }
        NEXT();
    }
op_srl: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = GPR(rs1, _li) >> (GPR(rs2, _li) & 0x1F);
        PC(_li) += 4;
    }
    NEXT();
}
op_sra: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (uint32_t)((int32_t)GPR(rs1, _li) >> (GPR(rs2, _li) & 0x1F));
        PC(_li) += 4;
    }
    NEXT();
}
op_slt: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = ((int32_t)GPR(rs1, _li) < (int32_t)GPR(rs2, _li)) ? 1 : 0;
        PC(_li) += 4;
    }
    NEXT();
}
op_sltu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (GPR(rs1, _li) < GPR(rs2, _li)) ? 1 : 0;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * RV32M multiply/divide
     * ============================================================ */
op_mulh: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (uint32_t)(((int64_t)(int32_t)GPR(rs1, _li) * (int64_t)(int32_t)GPR(rs2, _li)) >> 32);
        PC(_li) += 4;
    }
    NEXT();
}
op_mulhsu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (uint32_t)(((int64_t)(int32_t)GPR(rs1, _li) * (uint64_t)GPR(rs2, _li)) >> 32);
        PC(_li) += 4;
    }
    NEXT();
}
op_mulhu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = (uint32_t)(((uint64_t)GPR(rs1, _li) * (uint64_t)GPR(rs2, _li)) >> 32);
        PC(_li) += 4;
    }
    NEXT();
}
op_div: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        int32_t a = (int32_t)GPR(rs1, _li), b = (int32_t)GPR(rs2, _li);
        GPR(rd, _li) = b ? (uint32_t)((a == INT32_MIN && b == -1) ? INT32_MIN : a / b) : (uint32_t)-1;
        PC(_li) += 4;
    }
    NEXT();
}
op_divu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li), b = GPR(rs2, _li);
        GPR(rd, _li) = b ? a / b : 0xFFFFFFFF;
        PC(_li) += 4;
    }
    NEXT();
}
op_rem: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        int32_t a = (int32_t)GPR(rs1, _li), b = (int32_t)GPR(rs2, _li);
        GPR(rd, _li) = b ? (uint32_t)((a == INT32_MIN && b == -1) ? 0 : a % b) : GPR(rs1, _li);
        PC(_li) += 4;
    }
    NEXT();
}
op_remu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li), b = GPR(rs2, _li);
        GPR(rd, _li) = b ? a % b : a;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * CSR — 标量 lane 0 (CSR 本身是 per-hart，所有 lane 独立)
     * 通过 fcsr/mhartid 指针访问 (调度器提供 lane 0 的结构引用)
     * ============================================================ */
/* CSR 访问需要 lane 结构中的 fcsr/mhartid 字段。
     * 引擎不持有 GPGPULane*，所以使用 INLINE 简化的寄存器文件访问。
     * CSR 值存储在 gpr/fpr 之外的专用位置：引擎需要额外的 lane 状态指针。
     * 简化方案: 调度器传入 mhartid[] 和 fcsr[] 的 SoA 数组。
     *
     * CSR 只操作 lane 0 的 fcsr/mhartid，通过 SoA 数组 fcsr[]/mhartid[]。
     * 引擎需要额外的 lane 状态指针来做 CSR 操作。
     *
     * FIXME: 将 lane 元数据 (mhartid, fcsr, fp_status) 也 SoA 化。
     */
/* 暂时通过 ctx->s 和 warp lane 0 的引用做 CSR，过渡方案 */

/* ============================================================
     * ebreak / done
     * ============================================================ */
/* ============================================================
     * VPU 向量指令 — warp 级 SIMD (custom-1 opcode 0x2B)
     * ============================================================ */
#define VPU_TRACE 0
#if VPU_TRACE
#define VTRACE(fmt, ...) fprintf(stderr, "[VPU] " fmt, ##__VA_ARGS__)
#else
#define VTRACE(fmt, ...) ((void)0)
#endif
op_vld_v: {
    int vd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    uint32_t stride = GPR(rs2, 0); /* 标量 stride, lane 0 为准 */
    VTRACE("vld_v  v%d, (0x%x), r%d  active=0x%x\n", vd - 16, GPR(rs1, 0), rs2, _active);
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + _li * stride;
        FPR(vd, _li) = (LIKELY(a + 4 <= s->vram_size)) ? *(uint32_t *)(s->vram_ptr + a) : 0;
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 4;)
    NEXT();
}
op_vst_v: {
    int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, vs3 = ip[-1].rd;
    uint32_t stride = GPR(rs2, 0); /* 标量 stride, lane 0 为准 */
    VTRACE("vst_v  vs3=%d a=0x%x r%d  v2[0]=%.3f C[0]=%.3f\n", vs3, GPR(rs1, 0), rs2, FR(vs3, 0),
           *(float *)(s->vram_ptr + GPR(rs1, 0)));
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + _li * stride;
        if (LIKELY(a + 4 <= s->vram_size)) *(uint32_t *)(s->vram_ptr + a) = FPR(vs3, _li);
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 4;)
    NEXT();
}
#define VFALU(name, op)                                         \
    op_##name:                                                  \
    {                                                           \
        int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            FR(vd, _li) = FR(vs1, _li) op FR(vs2, _li);         \
            PC(_li) += 4;                                       \
        }                                                       \
        NEXT();                                                 \
    }
    VFALU(vfadd_v, +)
    VFALU(vfsub_v, -)
    VFALU(vfdiv_v, /)
#undef VFALU

op_vfmul_v: {
    int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
    VTRACE("vfmul_v vd=%d vs1=%d vs2=%d  v0=%.3f v1=%.3f\n", vd, vs1, vs2, FR(vs1, 0), FR(vs2, 0));
    FOR_EACH_LANE
    {
        FR(vd, _li) = FR(vs1, _li) * FR(vs2, _li);
        PC(_li) += 4;
    }
    VTRACE("vfmul_v result v2[0]=%.3f v2[1]=%.3f\n", FR(vd, 0), FR(vd, 1));
    NEXT();
}

op_vffma_v: {
    int vd = ip[-1].rd, vs1 = ip[-1].rs1, vs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        FR(vd, _li) += FR(vs1, _li) * FR(vs2, _li);
        PC(_li) += 4;
    }
    NEXT();
}
#define VFALUS(name, op)                      \
    op_##name:                                \
    {                                         \
        int vd = ip[-1].rd, vs1 = ip[-1].rs1; \
        float sc = FR(ip[-1].rs2, 0);         \
        FOR_EACH_LANE                         \
        {                                     \
            FR(vd, _li) = FR(vs1, _li) op sc; \
            PC(_li) += 4;                     \
        }                                     \
        NEXT();                               \
    }
    VFALUS(vfmul_vs, *)
    VFALUS(vfadd_vs, +)
    VFALUS(vfdiv_vs, /)
#undef VFALUS

#define VFUNA(name, expr)                     \
    op_##name:                                \
    {                                         \
        int vd = ip[-1].rd, vs1 = ip[-1].rs1; \
        FOR_EACH_LANE                         \
        {                                     \
            float v = FR(vs1, _li);           \
            FR(vd, _li) = expr;               \
            PC(_li) += 4;                     \
        }                                     \
        NEXT();                               \
    }
    VFUNA(vfexp_v, fastexp(v))
    VFUNA(vfsig_v, fastsigmoid(v))
    VFUNA(vftanh_v, fasttanh(v))
    VFUNA(vfsqrt_v, sqrtf(v))
#undef VFUNA

op_vredsum_v: {
    int rd = ip[-1].rd, vs1 = ip[-1].rs1;
    float sum = 0;
    FOR_EACH_LANE
    {
        sum += FR(vs1, _li);
        PC(_li) += 4;
    }
    FR(rd, 0) = sum;
    NEXT();
}
op_vredmax_v: {
    int rd = ip[-1].rd, vs1 = ip[-1].rs1;
    float mx = NAN;
    FOR_EACH_LANE
    {
        if (isnan(mx) || FR(vs1, _li) > mx) mx = FR(vs1, _li);
        PC(_li) += 4;
    }
    FR(rd, 0) = mx;
    NEXT();
}

    /* ============================================================
     * TCU 矩阵指令 — warp-MMA (custom-1 opcode 0x2B, funct3=111)
     * mma 配置是 warp 本地变量, 避免多 warp 并发写入 s->mma 的 data race
     * ============================================================ */
op_mma_cfg: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE PC(_li) += 4;
    NEXT();
}
op_mma_zero: {
    int acc = ip[-1].rd;
    memset(&_mma_acc[acc * 32], 0, 128);
    FOR_EACH_LANE PC(_li) += 4;
    NEXT();
}
op_mma_ld: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) _mma_acc[acc * 32 + _li] = *(float *)(s->vram_ptr + a);
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_s: {
    int acc = ip[-1].rd, vs1 = ip[-1].rs1, fs2 = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    float ak = FR(fs2, 0);
    FOR_EACH_LANE
    {
        _mma_acc[acc * 32 + _li] += FR(vs1, _li) * ak;
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_st: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) *(float *)(s->vram_ptr + a) = _mma_acc[acc * 32 + _li];
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_relu: {
    int acc = ip[-1].rd;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        float v = _mma_acc[acc * 32 + _li];
        _mma_acc[acc * 32 + _li] = v > 0 ? v : 0;
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_bias: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) _mma_acc[acc * 32 + _li] += *(float *)(s->vram_ptr + a);
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * RV32A 原子指令 — 串行模式退化为普通 RMW
     * ============================================================ */
op_lr_w: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li);
        GPR(rd, _li) = ctrl_read(ctx, a, _li); /* load exclusive = normal load */
        PC(_li) += 4;
    }
    NEXT();
}
op_sc_w: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li);
        gpu_write(s, a, 4, GPR(rs2, _li));
        GPR(rd, _li) = 0; /* always success */
        PC(_li) += 4;
    }
    NEXT();
}

#define AMO(name, op)                                           \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            uint32_t a = GPR(rs1, _li);                         \
            uint32_t old = ctrl_read(ctx, a, _li);              \
            gpu_write(s, a, 4, (old op GPR(rs2, _li)));         \
            GPR(rd, _li) = old;                                 \
            PC(_li) += 4;                                       \
        }                                                       \
        NEXT();                                                 \
    }
    AMO(amoadd_w, +)
#undef AMO

#define AMO2(name, expr)                                        \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            uint32_t a = GPR(rs1, _li), v = GPR(rs2, _li);      \
            uint32_t old = ctrl_read(ctx, a, _li);              \
            gpu_write(s, a, 4, (uint32_t)expr);                 \
            GPR(rd, _li) = old;                                 \
            PC(_li) += 4;                                       \
        }                                                       \
        NEXT();                                                 \
    }
    AMO2(amoxor_w, old ^ v)
    AMO2(amoand_w, old & v)
    AMO2(amoor_w, old | v)
    AMO2(amomin_w, ((int32_t)old < (int32_t)v) ? old : v)
    AMO2(amomax_w, ((int32_t)old > (int32_t)v) ? old : v)
    AMO2(amominu_w, (old < v) ? old : v)
    AMO2(amomaxu_w, (old > v) ? old : v)
#undef AMO2

/* amoswap has different pattern */
op_amoswap_w: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li), v = GPR(rs2, _li);
        uint32_t old = ctrl_read(ctx, a, _li);
        gpu_write(s, a, 4, v);
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * tex — bilinear texture fetch
     * rs1=u(float), rs2=v(float), rd=dest, imm=base offset in VRAM
     * 2x2 bilinear: lerp(lerp(s00,s10,u), lerp(s01,s11,u), v)
     * ============================================================ */
op_tex: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    uint32_t base = (uint32_t)ip[-1].imm;
    FOR_EACH_LANE
    {
        float u = fabsf(FR(rs1, _li)), v = fabsf(FR(rs2, _li));
        int iu = (int)u, iv = (int)v;
        /* texture is 256-wide; clamp coords to VRAM bounds */
        if (iu < 0) iu = 0;
        if (iu > 254) {
            iu = 254;
            s->error_status |= GPGPU_ERR_VRAM_FAULT;
        }
        int max_row = (int)((s->vram_size - base) / (256 * 4)) - 1;
        if (max_row < 1) max_row = 1;
        if (iv < 0) iv = 0;
        if (iv >= max_row) {
            iv = max_row - 1;
            s->error_status |= GPGPU_ERR_VRAM_FAULT;
        }
        float fu = u - (float)iu, fv = v - (float)iv;
        float *tex = (float *)(s->vram_ptr + base);
        float s00 = tex[iv * 256 + iu], s10 = tex[iv * 256 + iu + 1];
        float s01 = tex[(iv + 1) * 256 + iu], s11 = tex[(iv + 1) * 256 + iu + 1];
        FR(rd, _li) = (1 - fu) * (1 - fv) * s00 + fu * (1 - fv) * s10 + (1 - fu) * fv * s01 + fu * fv * s11;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * barrier — warp 同步点 (custom-0 指令 0x0000000B)
     *
     * 所有 lane 到达 barrier → 返回 1, 由 scheduler 协调 block 内 warps。
     * ============================================================ */
op_barrier: {
    s->simt.barrier_active = true;
    memcpy(_stk_ext, _stk, sizeof(_stk)); // write-back to external
    *_sdepth_ext = _sdepth;
    return (int)(ip - code) | 0x10000; // bit16=1: barrier reached, low16=resume pc
}

/* ============================================================
     * ebreak / done — SIMT stack 感知出口
     * ============================================================ */
op_ebreak: {
    _active = 0;
    if (_sdepth > 0) {
        _sdepth--;
        _active = _stk[_sdepth].mask;
        ip = &code[_stk[_sdepth].ft_idx];
        NEXT();
    }
    return 0;
}
op_done: {
    _active = 0;
    if (_sdepth > 0) {
        _sdepth--;
        _active = _stk[_sdepth].mask;
        ip = &code[_stk[_sdepth].ft_idx];
        NEXT();
    }
    return 0;
}

/* ============================================================
     * FP Load/Store — VRAM fast path
     * ============================================================ */
op_flw: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        FPR(rd, _li) = (LIKELY(a + 4 <= s->vram_size)) ? *(uint32_t *)(s->vram_ptr + a) : 0;
        PC(_li) += 4;
    }
    PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 4;)
    COALESCE(ip[-1].rs1, ip[-1].imm);
    NEXT();
}
op_fsw: {
    int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(rs1, _li) + imm;
        if (LIKELY(a + 4 <= s->vram_size)) *(uint32_t *)(s->vram_ptr + a) = FPR(rs2, _li);
    }
    FOR_EACH_LANE PC(_li) += 4;
    PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 4;)
    NEXT();
}

/* ============================================================
     * FP 基础算术 — 硬件 float，-O3 自动向量化为 vmulps/vaddps
     * UNLIKELY 冷路径: 仅在 frm≠RNE 时触发舍入 + fflags
     * ============================================================ */
#if ENABLE_FP_CSR
#define FPBIN(name, op, is_div)                                 \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            FP_BIN_CSR(rd, rs1, rs2, op, is_div);               \
        }                                                       \
        FOR_EACH_LANE PC(_li) += 4;                             \
        NEXT();                                                 \
    }
#else
#define FPBIN(name, op, is_div)                                 \
    op_##name:                                                  \
    {                                                           \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE                                           \
        {                                                       \
            FR(rd, _li) = FR(rs1, _li) op FR(rs2, _li);         \
        }                                                       \
        FOR_EACH_LANE PC(_li) += 4;                             \
        NEXT();                                                 \
    }
#endif
    FPBIN(fadd_s, +, 0)
    FPBIN(fsub_s, -, 0)
    FPBIN(fmul_s, *, 0)
    FPBIN(fdiv_s, /, 1)
#undef FPBIN

/* FMA — ENABLE_FP_CSR 控制 */
#if ENABLE_FP_CSR
#define FMA_HANDLER(name, expr)                                                                \
    op_##name:                                                                                 \
    {                                                                                          \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;              \
        FOR_EACH_LANE                                                                          \
        {                                                                                      \
            float _a = FR(rs1, _li), _b = FR(rs2, _li), _c = FR(rs3, _li);                     \
            float _r = (expr);                                                                 \
            if (UNLIKELY(FCSR_FRM(_fcsr[_li]) != 0)) _r = fpu_round(_r, FCSR_FRM(_fcsr[_li])); \
            if (UNLIKELY(isfinite(_r) == 0))                                                   \
                fpu_check_fflags(&_fcsr[_li], _r, _a *_b, _c, 0);                              \
            else                                                                               \
                _fcsr[_li] |= 0x01;                                                            \
            FR(rd, _li) = _r;                                                                  \
        }                                                                                      \
        FOR_EACH_LANE PC(_li) += 4;                                                            \
        NEXT();                                                                                \
    }
#else
#define FMA_HANDLER(name, expr)                                                   \
    op_##name:                                                                    \
    {                                                                             \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3; \
        FOR_EACH_LANE                                                             \
        {                                                                         \
            FR(rd, _li) = (expr);                                                 \
        }                                                                         \
        FOR_EACH_LANE PC(_li) += 4;                                               \
        NEXT();                                                                   \
    }
#endif
    FMA_HANDLER(fmadd_s, FR(rs1, _li) * FR(rs2, _li) + FR(rs3, _li))
    FMA_HANDLER(fmsub_s, FR(rs1, _li) * FR(rs2, _li) - FR(rs3, _li))
    FMA_HANDLER(fnmsub_s, -(FR(rs1, _li) * FR(rs2, _li) - FR(rs3, _li)))
    FMA_HANDLER(fnmadd_s, -(FR(rs1, _li) * FR(rs2, _li) + FR(rs3, _li)))
#undef FMA_HANDLER

/* fsqrt */
op_fsqrt_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
#if ENABLE_FP_CSR
    FOR_EACH_LANE
    {
        float _a = FR(rs1, _li);
        FP_UNA_CSR(rd, rs1, sqrtf(_a));
        PC(_li) += 4;
    }
#else
    FOR_EACH_LANE
    {
        FR(rd, _li) = sqrtf(FR(rs1, _li));
        PC(_li) += 4;
    }
#endif
    NEXT();
}

/* 符号注入 — 位操作 */
op_fsgnj_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        FPR(rd, _li) = (FPR(rs1, _li) & ~0x80000000) | (FPR(rs2, _li) & 0x80000000);
        PC(_li) += 4;
    }
    NEXT();
}
op_fsgnjn_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        FPR(rd, _li) = (FPR(rs1, _li) & ~0x80000000) | ((~FPR(rs2, _li)) & 0x80000000);
        PC(_li) += 4;
    }
    NEXT();
}
op_fsgnjx_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        FPR(rd, _li) = FPR(rs1, _li) ^ (FPR(rs2, _li) & 0x80000000);
        PC(_li) += 4;
    }
    NEXT();
}

/* 最值 — RISC-V: NaN 返回非 NaN 操作数, 任一 sNaN 设 NV */
op_fmin_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        float a = FR(rs1, _li), b = FR(rs2, _li);
        FR(rd, _li) = (a != a) ? b : (b != b) ? a : (a < b ? a : b);
        if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
        PC(_li) += 4;
    }
    NEXT();
}
op_fmax_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        float a = FR(rs1, _li), b = FR(rs2, _li);
        FR(rd, _li) = (a != a) ? b : (b != b) ? a : (a > b ? a : b);
        if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * FP 比较 (整数结果写 gpr), NaN 输入设 NV
     * ============================================================ */
op_feq_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        float a = FR(rs1, _li), b = FR(rs2, _li);
        GPR(rd, _li) = (a == b) ? 1 : 0;
        if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
        PC(_li) += 4;
    }
    NEXT();
}
op_flt_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        float a = FR(rs1, _li), b = FR(rs2, _li);
        GPR(rd, _li) = (a < b) ? 1 : 0;
        if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
        PC(_li) += 4;
    }
    NEXT();
}
op_fle_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    FOR_EACH_LANE
    {
        float a = FR(rs1, _li), b = FR(rs2, _li);
        GPR(rd, _li) = (a <= b) ? 1 : 0;
        if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * FP 转换, UNLIKELY 冷路径支持 frm + fflags
     * ============================================================ */
op_fcvt_s_w: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        FR(rd, _li) = (float)(int32_t)GPR(rs1, _li);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_s_wu: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        FR(rd, _li) = (float)GPR(rs1, _li);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_w_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        float f = FR(rs1, _li);
        uint32_t raw = FPR(rs1, _li);
        GPR(rd, _li) = (uint32_t)(int32_t)(((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF)
                                                   ? 0x7FFFFFFF
                                                   : (f >= 2147483648.0f ? 0x7FFFFFFF
                                                                         : (f < -2147483648.0f ? (int32_t)0x80000000
                                                                                               : (int32_t)f)));
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_wu_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        float f = FR(rs1, _li);
        uint32_t raw = FPR(rs1, _li);
        GPR(rd, _li) = ((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF)
                               ? 0xFFFFFFFF
                               : (f < 0.0f ? 0 : (f >= 4294967296.0f ? 0xFFFFFFFF : (uint32_t)f));
        PC(_li) += 4;
    }
    NEXT();
}

/* 数据移动 */
op_fmv_w_x: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        FPR(rd, _li) = GPR(rs1, _li);
        PC(_li) += 4;
    }
    NEXT();
}
op_fmv_x_w: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        GPR(rd, _li) = FPR(rs1, _li);
        PC(_li) += 4;
    }
    NEXT();
}

/* fclass */
op_fclass_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t b = FPR(rs1, _li), e = (b >> 23) & 0xFF, m = b & 0x7FFFFF, sgn = (b >> 31) & 1;
        int r = 0;
        if (e == 0xFF)
            r = m ? (sgn ? (1 << 9) : (1 << 8)) : (sgn ? (1 << 0) : (1 << 7));
        else if (e == 0)
            r = m ? (sgn ? (1 << 2) : (1 << 5)) : (sgn ? (1 << 3) : (1 << 4));
        else
            r = sgn ? (1 << 1) : (1 << 6);
        GPR(rd, _li) = (uint32_t)r;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * 科学计算 — sfu.h fast approximations (无 libm 调用)
     * ============================================================ */
#if ENABLE_FP_CSR
#define SCI(name, expr)                       \
    op_##name:                                \
    {                                         \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1; \
        FOR_EACH_LANE                         \
        {                                     \
            float v = FR(rs1, _li);           \
            FP_UNA_CSR(rd, rs1, (expr));      \
            PC(_li) += 4;                     \
        }                                     \
        NEXT();                               \
    }
#else
#define SCI(name, expr)                       \
    op_##name:                                \
    {                                         \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1; \
        FOR_EACH_LANE                         \
        {                                     \
            float v = FR(rs1, _li);           \
            FR(rd, _li) = (expr);             \
            PC(_li) += 4;                     \
        }                                     \
        NEXT();                               \
    }
#endif
    SCI(fexp_s, fastexp(v))
    SCI(fln_s, logf(v))   /* libm: 无快速近似 */
    SCI(frcp_s, 1.0f / v) /* HW fdiv */
    SCI(frsqrt_s, fastrsqrt(v))
    SCI(ftanh_s, fasttanh(v))
    SCI(fsigmoid_s, fastsigmoid(v))
    SCI(fsin_s, sinf(v)) /* libm: ML 极少用 */
    SCI(fcos_s, cosf(v))
#undef SCI

/* ============================================================
     * CSR — per-lane 标量 (每个 lane 有独立的 mhartid/fcsr)
     * imm 即 CSR 地址 (12-bit)
     * ============================================================ */
op_csrrw: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t old = 0;
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            FCSR(_li) = (FCSR(_li) & ~0x1F) | (GPR(rs1, _li) & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            FCSR(_li) = (FCSR(_li) & ~0xE0) | ((GPR(rs1, _li) & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            FCSR(_li) = GPR(rs1, _li);
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}
op_csrrs: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t old = 0, mask = GPR(rs1, _li);
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            if (rs1) FCSR(_li) |= (mask & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            if (rs1) FCSR(_li) |= ((mask & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            if (rs1) FCSR(_li) |= mask;
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}
op_csrrc: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t old = 0, mask = GPR(rs1, _li);
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            if (rs1) FCSR(_li) &= ~(mask & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            if (rs1) FCSR(_li) &= ~((mask & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            if (rs1) FCSR(_li) &= ~mask;
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}
op_csrrwi: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd;
    uint32_t zimm = (uint32_t)ip[-1].rs1; /* rs1 字段 = uimm5 */
    FOR_EACH_LANE
    {
        uint32_t old = 0;
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            FCSR(_li) = (FCSR(_li) & ~0x1F) | (zimm & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            FCSR(_li) = (FCSR(_li) & ~0xE0) | ((zimm & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            FCSR(_li) = zimm;
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}
op_csrrsi: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd;
    uint32_t zimm = (uint32_t)ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t old = 0;
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            if (ip[-1].rs1) FCSR(_li) |= (zimm & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            if (ip[-1].rs1) FCSR(_li) |= ((zimm & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            if (ip[-1].rs1) FCSR(_li) |= zimm;
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}
op_csrrci: {
    uint16_t csr = (uint16_t)ip[-1].imm;
    int rd = ip[-1].rd;
    uint32_t zimm = (uint32_t)ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint32_t old = 0;
        switch (csr) {
        case CSR_MHARTID:
            old = MHARTID(_li); /* read-only per RISC-V spec */
            break;
        case CSR_FFLAGS:
            old = FCSR(_li) & 0x1F;
            if (ip[-1].rs1) FCSR(_li) &= ~(zimm & 0x1F);
            break;
        case CSR_FRM:
            old = (FCSR(_li) >> 5) & 7;
            if (ip[-1].rs1) FCSR(_li) &= ~((zimm & 7) << 5);
            break;
        case CSR_FCSR:
            old = FCSR(_li);
            if (ip[-1].rs1) FCSR(_li) &= ~zimm;
            break;
        }
        GPR(rd, _li) = old;
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * LP 浮点转换 — per-lane (位域操作, lpfp 接受值类型)
     * bf16 在 fpr 的 bits[31:16], e4m3/e5m2/e2m1 在 bits[31:24]
     * ============================================================ */
op_fcvt_s_bf16: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint16_t bf = (uint16_t)(FPR(rs1, _li) >> 16);
        FPR(rd, _li) = bf16_to_f32(bf);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_bf16_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint16_t bf = f32_to_bf16(FPR(rs1, _li));
        FPR(rd, _li) = (uint32_t)bf << 16; /* 低 16 位清零, 不保留旧值 */
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_s_e4m3: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e4 = (uint8_t)(FPR(rs1, _li) >> 24);
        FPR(rd, _li) = e4m3_to_f32(e4);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_e4m3_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e4 = f32_to_e4m3(FPR(rs1, _li));
        FPR(rd, _li) = (uint32_t)e4 << 24; /* 低 24 位清零 */
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_s_e5m2: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e5 = (uint8_t)(FPR(rs1, _li) >> 24);
        FPR(rd, _li) = e5m2_to_f32(e5);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_e5m2_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e5 = f32_to_e5m2(FPR(rs1, _li));
        FPR(rd, _li) = (uint32_t)e5 << 24; /* 低 24 位清零 */
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_s_e2m1: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e2 = (uint8_t)(FPR(rs1, _li) >> 24);
        FPR(rd, _li) = e2m1_to_f32(e2);
        PC(_li) += 4;
    }
    NEXT();
}
op_fcvt_e2m1_s: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1;
    FOR_EACH_LANE
    {
        uint8_t e2 = f32_to_e2m1(FPR(rs1, _li));
        FPR(rd, _li) = (uint32_t)e2 << 24; /* 低 24 位清零 */
        PC(_li) += 4;
    }
    NEXT();
}

/* ============================================================
     * 错误出口
     * ============================================================ */
op_illegal: {
    return -1;
}
}
