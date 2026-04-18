#ifndef INST_H
#define INST_H

#include "qemu/osdep.h"
#include "debug.h"   // 引入调试宏系统

#ifdef DEBUG_OPCODE_TABLE
#define IF_DEBUG_OPCODE_TABLE(code) code
#else
#define IF_DEBUG_OPCODE_TABLE(code) ((void)0)
#endif

#ifdef DEBUG_INST
#define IF_DEBUG_INST(code) code
#else
#define IF_DEBUG_INST(code) ((void)0)
#endif

#define NUM_OF_INST 300
#define MATCH_EBREAK 0x00100073

/* Instruction Parsing Macros and tools */
static inline uint32_t pattern_to_mask(const char *pattern) {
    uint32_t mask = 0;
    const char *p = pattern;
    int bit = 31;
    
    while (*p && bit >= 0) {
        if (*p == '0' || *p == '1') {
            mask |= (1U << bit);
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    return mask;
}

static inline uint32_t pattern_to_match(const char *pattern) {
    uint32_t match = 0;
    const char *p = pattern;
    int bit = 31;
    
    while (*p && bit >= 0) {
        if (*p == '1') {
            match |= (1U << bit);
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
    }
    
    return match;
}

/* ======== Register ======== */

/* default */
#define G(i)        (l->gpr[ctx->i].u32)
#define F(i)        (l->fpr[ctx->i].u32)
/* other choice */
#define G_CF(i)    (l->gpr[ctx->i].f32)
#define G_I32(i)    (l->gpr[ctx->i].i32)
#define F_CF(i)     (l->fpr[ctx->i].f32)
#define F_I32(i)    (l->fpr[ctx->i].i32)
#define F_BF16(i)   (l->fpr[ctx->i].bf16)
#define F_E4M3(i)   (l->fpr[ctx->i].e4m3)
#define F_E5M2(i)   (l->fpr[ctx->i].e5m2)
#define F_E2M1(i)   (l->fpr[ctx->i].e2m1)

/* ======== Memory ======== */
#define Mw(addr, len, data) gpu_write(ctx->s, addr, len, data)
#define Mr(addr, len) gpu_read(ctx->s, addr, len)

/* ======== IMM ======== */
#define immI(i)   (SEXT(BITS(i, 31, 20), 12))
#define immU(i)   ((SEXT(BITS(i, 31, 12), 20) << 12))
#define immS(i)   ((SEXT(BITS(i, 31, 25), 7) << 5) | BITS(i, 11, 7))
#define immB(i)   ((SEXT(BITS(i, 31, 31), 1) << 12) | (BITS(i, 7, 7) << 11) | (BITS(i, 30, 25) << 5) | (BITS(i, 11, 8) << 1))
#define immJ(i)   ((SEXT(BITS(i, 31, 31), 1) << 20) | (BITS(i, 19, 12) << 12) | (BITS(i, 20, 20) << 11) | (BITS(i, 30, 21) << 1))
#define immCSR(i) (BITS(inst, 31, 20))

/* ============== Context ================= */

/* context for int inst */
#define INIT_LANE_CONTEXT_IN() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    uint32_t old_pc = l->pc; \
    uint32_t src1 = G(rs1); \
    uint32_t src2 = G(rs2); \
    int32_t imm = ctx->imm; \
    (void)src1; (void)src2; (void)imm;

/* context for float inst */
#define INIT_LANE_CONTEXT_FP() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    uint32_t old_pc = l->pc; \
    float32 src1 = F(rs1); \
    float32 src2 = F(rs2); \
    float32 src3 = F(rs3); \
    int32_t imm = ctx->imm; \
    (void)src1; (void)src2; (void)src3; (void)imm;

/* minimum context for special inst */
#define INIT_LANE_CONTEXT_NO() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    int32_t imm = ctx->imm; \
    (void)imm;

/* 判断pc是否跳转 */
#define FINISH_LANE_CONTEXT() \
    if (l->pc == old_pc) l->pc += 4;

/* ================ Func Wrapper ================ */

/* for int inst */
#define EXEC_FUNC_IN(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_IN(); \
        IF_DEBUG_INST( \
            LogTrace("[EXEC] %-10s lane=%d pc=0x%08x rs1=%-2d(0x%08x) rs2=%-2d(0x%08x) imm=%d", \
                     #name, lane_id, old_pc, ctx->rs1, src1, ctx->rs2, src2, imm); \
        ) \
        code \
        IF_DEBUG_INST( \
            LogTrace("[RES ] %-10s lane=%d rd=%-2d val=0x%08x (%d)", \
                     #name, lane_id, ctx->rd, G(rd), G_I32(rd)); \
        ) \
        FINISH_LANE_CONTEXT(); \
    }

/* for float inst */
#define EXEC_FUNC_FP(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_FP(); \
        IF_DEBUG_INST( \
            LogTrace("[EXEC] %-10s lane=%d pc=0x%08x rs1=%-2d(0x%08x) rs2=%-2d(0x%08x) rs3=%-2d(0x%08x) imm=%d", \
                     #name, lane_id, old_pc, ctx->rs1, F(rs1), ctx->rs2, F(rs2), ctx->rs3, F(rs3), imm); \
        ) \
        /* sync_fcsr_to_fp_status */ \
        do { \
            uint8_t __frm = (l->fcsr >> 5) & 0x7; \
            if (__frm <= 4) { \
                l->fp_status.float_rounding_mode = __frm; \
            } else { \
                l->fp_status.float_rounding_mode = 0; /* RNE */ \
            } \
            l->fp_status.float_exception_flags = 0; \
        } while(0); \
        code \
        /* sync_fp_status_to_fcsr */ \
        do { \
            uint8_t __fflags = l->fp_status.float_exception_flags & 0x1F; \
            l->fcsr = (l->fcsr & ~0x1F) | __fflags; \
        } while(0); \
        IF_DEBUG_INST( \
            LogTrace("[RES ] %-10s lane=%d rd=%-2d val=0x%08x (%f)", \
                     #name, lane_id, ctx->rd, F(rd), F_CF(rd)); \
        ) \
        FINISH_LANE_CONTEXT(); \
    }

/* for special inst */
#define EXEC_FUNC_NO(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_NO(); \
        IF_DEBUG_INST( \
            LogTrace("[EXEC] %-10s lane=%d imm=%d", #name, lane_id, imm); \
        ) \
        code \
        IF_DEBUG_INST( \
            LogTrace("[RES ] %-10s lane=%d", #name, lane_id); \
        ) \
        FINISH_LANE_CONTEXT(); \
    }

#endif /* INST_H */
