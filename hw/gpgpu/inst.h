#ifndef INST_H
#define INST_H

#include "qemu/osdep.h"

#define NUM_OF_INST 50
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
#define F(i)        (l->fpr[ctx->i].f32)
/* other choice */
#define G_F32(i)    (l->gpr[ctx->i].f32)
#define G_I32(i)    (l->gpr[ctx->i].i32)
#define F_U32(i)    (l->fpr[ctx->i].u32)
#define F_I32(i)    (l->fpr[ctx->i].i32)
#define F_BF16(i)   (l->fpr[ctx->i].bf16)
#define F_E4M3(i)   (l->fpr[ctx->i].e4m3)
#define F_E5M2(i)   (l->fpr[ctx->i].e5m2)
#define F_E2M1(i)   (l->fpr[ctx->i].e2m1)

/* ======== Memory ======== */
#define Mw(addr, len, data) vram_write(ctx->s, addr, len, data)
#define Mr(addr, len) vram_read(ctx->s, addr, len)

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
    uint32_t src1 = G(rs1); \
    uint32_t src2 = G(rs2); \
    int32_t imm = ctx->imm; \
    (void)src1; (void)src2; (void)imm;

/* context for float inst */
#define INIT_LANE_CONTEXT_FP() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    float src1 = F(rs1); \
    float src2 = F(rs2); \
    float src3 = F(rs3); \
    int32_t imm = ctx->imm; \
    (void)src1; (void)src2; (void)src3; (void)imm;

/* minimum context for special inst */
#define INIT_LANE_CONTEXT_NO() \
    GPGPULane *l = &ctx->warp->lanes[lane_id]; \
    int32_t imm = ctx->imm; \
    (void)imm;

/* ================ Func Wrapper ================ */

/* for int inst */
#define EXEC_FUNC_IN(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_IN(); \
        code \
    }

/* for float inst */
#define EXEC_FUNC_FP(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_FP(); \
        code \
    }

/* for special inst */
#define EXEC_FUNC_NO(name, code) \
    static void __attribute__((unused)) exec_##name(exec_ctx_t *ctx, int lane_id) { \
        INIT_LANE_CONTEXT_NO(); \
        code \
    }

#endif /* INST_H */
