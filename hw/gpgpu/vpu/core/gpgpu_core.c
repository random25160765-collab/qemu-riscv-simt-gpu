/*
 * VPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "gpgpu.h"
#include "gpgpu_core.h"
#include "lpfp.h"
#include "memory.h"
#include "inst.h"
#include "utils.h"
#include "proto.h"

/* Define types of instruction */
typedef enum {
    TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR,
    TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4,
} inst_type_t;

/* context of warp */
typedef struct exec_ctx {
    GPGPUState *s;
    GPGPUWarp *warp;
    int rd; int rs1; int rs2; int rs3;
    int32_t imm;
    int type;
} exec_ctx_t;

static void get_warp_ctx(exec_ctx_t *ctx, uint32_t inst, int type)
{
    ctx->rd  = BITS(inst, 11, 7);
    ctx->rs1 = BITS(inst, 19, 15);
    ctx->rs2 = BITS(inst, 24, 20);
    ctx->rs3 = BITS(inst, 31, 27);

    switch (type) {
        case TYPE_I: case TYPE_FI:
            ctx->imm = immI(inst); break;
        case TYPE_S: case TYPE_FS:
            ctx->imm = immS(inst); break;
        case TYPE_U:
            ctx->imm = immU(inst); break;
        case TYPE_B:
            ctx->imm = immB(inst); break;
        case TYPE_J:
            ctx->imm = immJ(inst); break;
        case TYPE_R: case TYPE_FR: case TYPE_F4:
            ctx->imm = 0; break;
        case TYPE_CSR:
            ctx->imm = immCSR(inst); break;
        default:
            ctx->imm = 0; break;
    }
}

/* CSR 读取函数 */
static uint32_t csr_read(GPGPULane *l, uint16_t csr_addr) {
    switch (csr_addr) {
        case CSR_MHARTID:
            return l->mhartid;
        case CSR_FFLAGS:
            return l->fcsr & 0x1F;
        case CSR_FRM:
            return (l->fcsr >> 5) & 0x7;
        case CSR_FCSR:
            return l->fcsr;
        default:
            VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_CSR_ACCESS, csr_addr);
            return 0;
    }
}

/* CSR 写入函数 */
static void csr_write(GPGPULane *l, uint16_t csr_addr, uint32_t val) {
    switch (csr_addr) {
        case CSR_FFLAGS:
            l->fcsr = (l->fcsr & ~0x1F) | (val & 0x1F);
            break;
        case CSR_FRM:
            l->fcsr = (l->fcsr & ~0xE0) | ((val & 0x7) << 5);
            break;
        case CSR_FCSR:
            l->fcsr = val;
            break;
        case CSR_MHARTID:
            VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_CSR_ACCESS, csr_addr);
            break;
        default:
            VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_CSR_ACCESS, csr_addr);
            break;
    }
}

/* ======================================= Function Table =============================================== */

/* ======== RV32I ======== */

/* 1. control and branch inst */
EXEC_FUNC_IN(jal,      {
			GPGPU_INST_BIN(INST_JAL, (uint32_t)rd, (uint32_t)imm, l->pc);
			G(rd) = l->pc + 4; l->pc += imm;
	})
EXEC_FUNC_IN(jalr,     {
			GPGPU_INST_BIN(INST_JALR, (uint32_t)rd, src1, (uint32_t)imm, l->pc);
			G(rd) = l->pc + 4; l->pc = (src1 + imm) & ~1;
	})
EXEC_FUNC_IN(beq,      {
			GPGPU_INST_BIN(INST_BEQ, src1, src2, (uint32_t)imm, l->pc);
			if (src1 == src2) l->pc += imm;
	})
EXEC_FUNC_IN(bne,      {
			GPGPU_INST_BIN(INST_BNE, src1, src2, (uint32_t)imm, l->pc);
			if (src1 != src2) l->pc += imm;
	})
EXEC_FUNC_IN(blt,      {
			GPGPU_INST_BIN(INST_BLT, src1, src2, (uint32_t)imm, l->pc);
			if ((int32_t)src1 < (int32_t)src2) l->pc += imm;
	})
EXEC_FUNC_IN(bge,      {
			GPGPU_INST_BIN(INST_BGE, src1, src2, (uint32_t)imm, l->pc);
			if ((int32_t)src1 >= (int32_t)src2) l->pc += imm;
	})
EXEC_FUNC_IN(bltu,     {
			GPGPU_INST_BIN(INST_BLTU, src1, src2, (uint32_t)imm, l->pc);
			if (src1 < src2) l->pc += imm;
	})
EXEC_FUNC_IN(bgeu,     {
			GPGPU_INST_BIN(INST_BGEU, src1, src2, (uint32_t)imm, l->pc);
			if (src1 >= src2) l->pc += imm;
	})

/* 2. memory IO inst */
EXEC_FUNC_IN(lb,       {
			uint32_t addr = src1 + imm;
			uint8_t val = Mr(addr, 1);
			GPGPU_INST_BIN(INST_LB, (uint32_t)rd, addr);
			G_I32(rd) = (int32_t)(val << 24) >> 24;
	})
EXEC_FUNC_IN(lh,       {
			uint32_t addr = src1 + imm;
			uint16_t val = Mr(addr, 2);
			GPGPU_INST_BIN(INST_LH, (uint32_t)rd, addr);
			G_I32(rd) = (int32_t)(val << 16) >> 16;
	})
EXEC_FUNC_IN(lw,       {
			uint32_t addr = src1 + imm;
			uint32_t val = Mr(addr, 4);
			GPGPU_INST_BIN(INST_LW, (uint32_t)rd, addr);
			G(rd) = val;
	})
EXEC_FUNC_IN(lbu,      {
			uint32_t addr = src1 + imm;
			uint8_t val = Mr(addr, 1);
			GPGPU_INST_BIN(INST_LBU, (uint32_t)rd, addr);
			G(rd) = val;
	})
EXEC_FUNC_IN(lhu,      {
			uint32_t addr = src1 + imm;
			uint16_t val = Mr(addr, 2);
			GPGPU_INST_BIN(INST_LHU, (uint32_t)rd, addr);
			G(rd) = val;
	})
EXEC_FUNC_IN(sb,       {
			uint32_t addr = src1 + imm;
			GPGPU_INST_BIN(INST_SB, addr, src2);
			Mw(addr, 1, src2);
	})
EXEC_FUNC_IN(sh,       {
			uint32_t addr = src1 + imm;
			GPGPU_INST_BIN(INST_SH, addr, src2);
			Mw(addr, 2, src2);
	})
EXEC_FUNC_IN(sw,       {
			uint32_t addr = src1 + imm;
			GPGPU_INST_BIN(INST_SW, addr, src2);
			Mw(addr, 4, src2);
	})

/* 3. unary integer operator */
EXEC_FUNC_IN(lui,      {
			GPGPU_INST_BIN(INST_LUI, (uint32_t)rd, (uint32_t)imm);
			G(rd) = imm;
	})
EXEC_FUNC_IN(auipc,    {
			GPGPU_INST_BIN(INST_AUIPC, (uint32_t)rd, (uint32_t)imm, l->pc);
			G(rd) = l->pc + imm;
	})
EXEC_FUNC_IN(addi,     {
			GPGPU_INST_BIN(INST_ADDI, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = src1 + imm;
	})
EXEC_FUNC_IN(slti,     {
			GPGPU_INST_BIN(INST_SLTI, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = ((int32_t)src1 < imm) ? 1 : 0;
	})
EXEC_FUNC_IN(sltiu,    {
			GPGPU_INST_BIN(INST_SLTIU, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = (src1 < (uint32_t)imm) ? 1 : 0;
	})
EXEC_FUNC_IN(xori,     {
			GPGPU_INST_BIN(INST_XORI, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = src1 ^ imm;
	})
EXEC_FUNC_IN(ori,      {
			GPGPU_INST_BIN(INST_ORI, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = src1 | imm;
	})
EXEC_FUNC_IN(andi,     {
			GPGPU_INST_BIN(INST_ANDI, (uint32_t)rd, src1, (uint32_t)imm);
			G(rd) = src1 & imm;
	})
EXEC_FUNC_IN(slli,     {
			GPGPU_INST_BIN(INST_SLLI, (uint32_t)rd, src1, imm & 0x1F);
			G(rd) = src1 << (imm & 0x1F);
	})
EXEC_FUNC_IN(srli,     {
			GPGPU_INST_BIN(INST_SRLI, (uint32_t)rd, src1, imm & 0x1F);
			G(rd) = src1 >> (imm & 0x1F);
	})
EXEC_FUNC_IN(srai,     {
			GPGPU_INST_BIN(INST_SRAI, (uint32_t)rd, src1, imm & 0x1F);
			G_I32(rd) = (int32_t)src1 >> (imm & 0x1F);
	})

/* 4. binary integer operator */
EXEC_FUNC_IN(add,      {
			GPGPU_INST_BIN(INST_ADD, (uint32_t)rd, src1, src2);
			G(rd) = src1 + src2;
	})
EXEC_FUNC_IN(sub,      {
			GPGPU_INST_BIN(INST_SUB, (uint32_t)rd, src1, src2);
			G(rd) = src1 - src2;
	})
EXEC_FUNC_IN(sll,      {
			GPGPU_INST_BIN(INST_SLL, (uint32_t)rd, src1, src2 & 0x1F);
			G(rd) = src1 << (src2 & 0x1F);
	})
EXEC_FUNC_IN(slt,      {
			GPGPU_INST_BIN(INST_SLT, (uint32_t)rd, src1, src2);
			G(rd) = ((int32_t)src1 < (int32_t)src2) ? 1 : 0;
	})
EXEC_FUNC_IN(sltu,     {
			GPGPU_INST_BIN(INST_SLTU, (uint32_t)rd, src1, src2);
			G(rd) = (src1 < src2) ? 1 : 0;
	})
EXEC_FUNC_IN(xor,      {
			GPGPU_INST_BIN(INST_XOR, (uint32_t)rd, src1, src2);
			G(rd) = src1 ^ src2;
	})
EXEC_FUNC_IN(srl,      {
			GPGPU_INST_BIN(INST_SRL, (uint32_t)rd, src1, src2 & 0x1F);
			G(rd) = src1 >> (src2 & 0x1F);
	})
EXEC_FUNC_IN(sra,      {
			GPGPU_INST_BIN(INST_SRA, (uint32_t)rd, src1, src2 & 0x1F);
			G_I32(rd) = (int32_t)src1 >> (src2 & 0x1F);
	})
EXEC_FUNC_IN(or,       {
			GPGPU_INST_BIN(INST_OR, (uint32_t)rd, src1, src2);
			G(rd) = src1 | src2;
	})
EXEC_FUNC_IN(and,      {
			GPGPU_INST_BIN(INST_AND, (uint32_t)rd, src1, src2);
			G(rd) = src1 & src2;
	})

/* 5. RV32M mul & div expansion */
EXEC_FUNC_IN(mul,      {
			GPGPU_INST_BIN(INST_MUL, (uint32_t)rd, src1, src2);
			G(rd) = src1 * src2;
	})
EXEC_FUNC_IN(mulh,     {
			GPGPU_INST_BIN(INST_MULH, (uint32_t)rd, src1, src2);
			G(rd) = (uint32_t)(((int64_t)(int32_t)src1 * (int64_t)(int32_t)src2) >> 32);
	})
EXEC_FUNC_IN(mulhsu,   {
			GPGPU_INST_BIN(INST_MULHSU, (uint32_t)rd, src1, src2);
			G(rd) = (uint32_t)(((int64_t)(int32_t)src1 * (uint64_t)src2) >> 32);
	})
EXEC_FUNC_IN(mulhu,    {
			GPGPU_INST_BIN(INST_MULHU, (uint32_t)rd, src1, src2);
			G(rd) = (uint32_t)(((uint64_t)src1 * (uint64_t)src2) >> 32);
	})
EXEC_FUNC_IN(div,      {
			GPGPU_INST_BIN(INST_DIV, (uint32_t)rd, src1, src2);
			G_I32(rd) = (src2 == 0) ? -1 : (int32_t)src1 / (int32_t)src2;
	})
EXEC_FUNC_IN(divu,     {
			GPGPU_INST_BIN(INST_DIVU, (uint32_t)rd, src1, src2);
			G(rd) = (src2 == 0) ? 0xFFFFFFFF : src1 / src2;
	})
EXEC_FUNC_IN(rem,      {
			GPGPU_INST_BIN(INST_REM, (uint32_t)rd, src1, src2);
			G_I32(rd) = (src2 == 0) ? src1 : (int32_t)src1 % (int32_t)src2;
	})
EXEC_FUNC_IN(remu,     {
			GPGPU_INST_BIN(INST_REMU, (uint32_t)rd, src1, src2);
			G(rd) = (src2 == 0) ? src1 : src1 % src2;
	})

/* 6. system inst */
EXEC_FUNC_IN(ebreak,   {
			GPGPU_INST_BIN(INST_EBREAK);
			/* nothing */
	})

/* CSRRW */
EXEC_FUNC_IN(csrrw, {
	    uint16_t csr = (uint16_t)imm;
	    uint32_t old_val = csr_read(l, csr);
	    GPGPU_INST_BIN(INST_CSRRW, (uint32_t)rd, (uint32_t)csr, src1);
	    G(rd) = old_val;
	    csr_write(l, csr, src1);
	})

/* CSRRS */
EXEC_FUNC_IN(csrrs, {
	    uint16_t csr = (uint16_t)imm;
	    uint32_t old_val = csr_read(l, csr);
	    GPGPU_INST_BIN(INST_CSRRS, (uint32_t)rd, (uint32_t)csr, src1);
	    G(rd) = old_val;
	    if (ctx->rs1 != 0) {
	        csr_write(l, csr, old_val | src1);
	    }
	})

/* CSRRC */
EXEC_FUNC_IN(csrrc, {
	    uint16_t csr = (uint16_t)imm;
	    uint32_t old_val = csr_read(l, csr);
	    GPGPU_INST_BIN(INST_CSRRC, (uint32_t)rd, (uint32_t)csr, src1);
	    G(rd) = old_val;
	    if (ctx->rs1 != 0) {
	        csr_write(l, csr, old_val & ~src1);
	    }
	})

/* CSRRWI */
EXEC_FUNC_IN(csrrwi, {
	    uint16_t csr = (uint16_t)imm;
	    GPGPU_INST_BIN(INST_CSRRWI, (uint32_t)rd, (uint32_t)csr, (uint32_t)ctx->rs1);
	    uint32_t old_val = csr_read(l, csr);
	    G(rd) = old_val;
	    csr_write(l, csr, (uint32_t)ctx->rs1);
	})

/* CSRRSI */
EXEC_FUNC_IN(csrrsi, {
	    uint16_t csr = (uint16_t)imm;
	    GPGPU_INST_BIN(INST_CSRRSI, (uint32_t)rd, (uint32_t)csr, (uint32_t)ctx->rs1);
	    uint32_t old_val = csr_read(l, csr);
	    G(rd) = old_val;
	    if (ctx->rs1 != 0) {
	        csr_write(l, csr, old_val | (uint32_t)ctx->rs1);
	    }
	})

/* CSRRCI */
EXEC_FUNC_IN(csrrci, {
	    uint16_t csr = (uint16_t)imm;
	    uint32_t old_val = csr_read(l, csr);
	    GPGPU_INST_BIN(INST_CSRRCI, (uint32_t)rd, (uint32_t)csr, (uint32_t)imm);
	    G(rd) = old_val;
	    if (ctx->rs1 != 0) {
	        csr_write(l, csr, old_val & ~(uint32_t)ctx->rs1);
	    }
	})

/* ============ RV32F ============ */

/* 访存指令（特殊实现） */
static void __attribute__((unused)) exec_flw(exec_ctx_t *ctx, int lane_id) {
	    GPGPULane *l = &ctx->warp->lanes[lane_id];
	    uint32_t old_pc = l->pc;
	    uint32_t addr = G(rs1) + ctx->imm;

	    GPGPU_INST_BIN(INST_FLW, (uint32_t)ctx->rd, addr);
	    uint32_t val = Mr(addr, 4);
	    F(rd) = val;

	    if (l->pc == old_pc) l->pc += 4;
	}

static void __attribute__((unused)) exec_fsw(exec_ctx_t *ctx, int lane_id) {
	    GPGPULane *l = &ctx->warp->lanes[lane_id];
	    uint32_t old_pc = l->pc;
	    uint32_t addr = G(rs1) + ctx->imm;

	    uint32_t val = F(rs2);
	    GPGPU_INST_BIN(INST_FSW, addr, val);
	    Mw(addr, 4, val);

	    if (l->pc == old_pc) l->pc += 4;
	}

/* 符号注入 */
EXEC_FUNC_FP(fsgnj_s,  {
	    GPGPU_INST_BIN(INST_FSGNJ_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = (F(rs1) & ~0x80000000) | (F(rs2) & 0x80000000);
	})
EXEC_FUNC_FP(fsgnjn_s, {
	    GPGPU_INST_BIN(INST_FSGNJN_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = (F(rs1) & ~0x80000000) | ((~F(rs2)) & 0x80000000);
	})
EXEC_FUNC_FP(fsgnjx_s, {
	    GPGPU_INST_BIN(INST_FSGNJX_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = F(rs1) ^ (F(rs2) & 0x80000000);
	})

/* 算术运算 */
EXEC_FUNC_FP(fadd_s,   {
	    GPGPU_INST_BIN(INST_FADD_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_add(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fsub_s,   {
	    GPGPU_INST_BIN(INST_FSUB_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_sub(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fmul_s,   {
	    GPGPU_INST_BIN(INST_FMUL_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_mul(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fdiv_s,   {
	    GPGPU_INST_BIN(INST_FDIV_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_div(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fsqrt_s,  {
	    GPGPU_INST_BIN(INST_FSQRT_S, (uint32_t)rd, F(rs1));
	    F(rd) = float32_sqrt(F(rs1), &l->fp_status);
	})

/* 乘加指令 */
EXEC_FUNC_FP(fmadd_s,  {
	    GPGPU_INST_BIN(INST_FMADD_S, (uint32_t)rd, F(rs1), F(rs2), F(rs3));
	    F(rd) = float32_muladd(F(rs1), F(rs2), F(rs3), 0, &l->fp_status);
	})
EXEC_FUNC_FP(fmsub_s,  {
	    GPGPU_INST_BIN(INST_FMSUB_S, (uint32_t)rd, F(rs1), F(rs2), F(rs3));
	    F(rd) = float32_muladd(F(rs1), F(rs2), F(rs3), float_muladd_negate_c, &l->fp_status);
	})
EXEC_FUNC_FP(fnmsub_s, {
	    GPGPU_INST_BIN(INST_FNMSUB_S, (uint32_t)rd, F(rs1), F(rs2), F(rs3));
	    F(rd) = float32_muladd(F(rs1), F(rs2), F(rs3), float_muladd_negate_product, &l->fp_status);
	})
EXEC_FUNC_FP(fnmadd_s, {
	    GPGPU_INST_BIN(INST_FNMADD_S, (uint32_t)rd, F(rs1), F(rs2), F(rs3));
	    F(rd) = float32_muladd(F(rs1), F(rs2), F(rs3), float_muladd_negate_result, &l->fp_status);
	})

/* 最值 */
EXEC_FUNC_FP(fmin_s,   {
	    GPGPU_INST_BIN(INST_FMIN_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_min(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fmax_s,   {
	    GPGPU_INST_BIN(INST_FMAX_S, (uint32_t)rd, F(rs1), F(rs2));
	    F(rd) = float32_max(F(rs1), F(rs2), &l->fp_status);
	})

/* 转换指令 */
EXEC_FUNC_FP(fcvt_s_w,   {
	    GPGPU_INST_BIN(INST_FCVT_S_W, (uint32_t)rd, G_I32(rs1));
	    F(rd) = int32_to_float32(G_I32(rs1), &l->fp_status);
	})
EXEC_FUNC_FP(fcvt_s_wu,  {
	    GPGPU_INST_BIN(INST_FCVT_S_WU, (uint32_t)rd, G(rs1));
	    F(rd) = uint32_to_float32(G(rs1), &l->fp_status);
	})

EXEC_FUNC_FP(fcvt_w_s, {
	    GPGPU_INST_BIN(INST_FCVT_W_S, (uint32_t)rd, F(rs1));
	    float32 f = F(rs1);
	    if (float32_is_quiet_nan(f, &l->fp_status) || float32_is_signaling_nan(f, &l->fp_status)) {
	        G_I32(rd) = 0x7FFFFFFF;
	    } else {
	        float32 max_int = int32_to_float32(0x7FFFFFFF, &l->fp_status);
	        float32 min_int = int32_to_float32(0x80000000, &l->fp_status);

	        if (float32_le(max_int, f, &l->fp_status)) {
	            G_I32(rd) = 0x7FFFFFFF;
	        } else if (float32_lt(f, min_int, &l->fp_status)) {
	            G_I32(rd) = 0x80000000;
	        } else {
	            G_I32(rd) = float32_to_int32(f, &l->fp_status);
	        }
	    }
	})

EXEC_FUNC_FP(fcvt_wu_s, {
	    GPGPU_INST_BIN(INST_FCVT_WU_S, (uint32_t)rd, F(rs1));
	    float32 f = F(rs1);
	    if (float32_is_quiet_nan(f, &l->fp_status) || float32_is_signaling_nan(f, &l->fp_status)) {
	        G(rd) = 0xFFFFFFFF;
	    } else if (float32_lt(f, 0, &l->fp_status)) {
	        G(rd) = 0;
	    } else {
	        float32 max_uint = uint32_to_float32(0xFFFFFFFF, &l->fp_status);
	        if (float32_le(max_uint, f, &l->fp_status)) {
	            G(rd) = 0xFFFFFFFF;
	        } else {
	            G(rd) = float32_to_uint32(f, &l->fp_status);
	        }
	    }
	})

/* 比较 */
EXEC_FUNC_FP(feq_s,   {
	    GPGPU_INST_BIN(INST_FEQ_S, (uint32_t)rd, F(rs1), F(rs2));
	    G(rd) = float32_eq(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(flt_s,   {
	    GPGPU_INST_BIN(INST_FLT_S, (uint32_t)rd, F(rs1), F(rs2));
	    G(rd) = float32_lt(F(rs1), F(rs2), &l->fp_status);
	})
EXEC_FUNC_FP(fle_s,   {
	    GPGPU_INST_BIN(INST_FLE_S, (uint32_t)rd, F(rs1), F(rs2));
	    G(rd) = float32_le(F(rs1), F(rs2), &l->fp_status);
	})

/* 数据移动 */
EXEC_FUNC_FP(fmv_w_x,  {
	    GPGPU_INST_BIN(INST_FMV_W_X, (uint32_t)rd, G(rs1));
	    F(rd) = G(rs1);
	})
EXEC_FUNC_FP(fmv_x_w,  {
	    GPGPU_INST_BIN(INST_FMV_X_W, (uint32_t)rd, F(rs1));
	    G(rd) = F(rs1);
	})

/* 分类 */
EXEC_FUNC_FP(fclass_s, {
	    GPGPU_INST_BIN(INST_FCLASS_S, (uint32_t)rd, F(rs1));
	    uint32_t bits = F(rs1);
	    uint32_t exp = (bits >> 23) & 0xFF;
	    uint32_t mant = bits & 0x7FFFFF;
	    uint32_t sign = (bits >> 31) & 1;
	    int result = 0;
	    if (exp == 0xFF) {
	        if (mant == 0) result = sign ? (1 << 0) : (1 << 7);
	        else result = sign ? (1 << 9) : (1 << 8);
	    } else if (exp == 0) {
	        if (mant == 0) result = sign ? (1 << 3) : (1 << 4);
	        else result = sign ? (1 << 2) : (1 << 5);
	    } else {
	        result = sign ? (1 << 1) : (1 << 6);
	    }
	    G(rd) = result;
	})

/* ======== LP float inst ======== */

EXEC_FUNC_FP(fcvt_s_bf16, {
	    GPGPU_INST_BIN(INST_FCVT_S_BF16, (uint32_t)rd, F_BF16(rs1));
	    F(rd) = bf16_to_f32(F_BF16(rs1));
	})
EXEC_FUNC_FP(fcvt_bf16_s, {
	    GPGPU_INST_BIN(INST_FCVT_BF16_S, (uint32_t)rd, F(rs1));
	    F_BF16(rd) = f32_to_bf16(F(rs1));
	})
EXEC_FUNC_FP(fcvt_s_e4m3, {
	    GPGPU_INST_BIN(INST_FCVT_S_E4M3, (uint32_t)rd, F_E4M3(rs1));
	    F(rd) = e4m3_to_f32(F_E4M3(rs1));
	})
EXEC_FUNC_FP(fcvt_e4m3_s, {
	    GPGPU_INST_BIN(INST_FCVT_E4M3_S, (uint32_t)rd, F(rs1));
	    F_E4M3(rd) = f32_to_e4m3(F(rs1));
	})
EXEC_FUNC_FP(fcvt_s_e5m2, {
	    GPGPU_INST_BIN(INST_FCVT_S_E5M2, (uint32_t)rd, F_E5M2(rs1));
	    F(rd) = e5m2_to_f32(F_E5M2(rs1));
	})
EXEC_FUNC_FP(fcvt_e5m2_s, {
	    GPGPU_INST_BIN(INST_FCVT_E5M2_S, (uint32_t)rd, F(rs1));
	    F_E5M2(rd) = f32_to_e5m2(F(rs1));
	})
EXEC_FUNC_FP(fcvt_s_e2m1, {
	    GPGPU_INST_BIN(INST_FCVT_S_E2M1, (uint32_t)rd, F_E2M1(rs1));
	    F(rd) = e2m1_to_f32(F_E2M1(rs1));
	})
EXEC_FUNC_FP(fcvt_e2m1_s, {
	    GPGPU_INST_BIN(INST_FCVT_E2M1_S, (uint32_t)rd, F(rs1));
	    F_E2M1(rd) = f32_to_e2m1(F(rs1));
	})

/* ======== Scientific functions ======== */

EXEC_FUNC_FP(fexp_s, {
	    GPGPU_INST_BIN(INST_FEXP_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = expf(v.f);
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(fln_s, {
	    GPGPU_INST_BIN(INST_FLN_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = logf(v.f);
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(frcp_s, {
	    GPGPU_INST_BIN(INST_FRCP_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = 1.0f / v.f;
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(frsqrt_s, {
	    GPGPU_INST_BIN(INST_FRSQRT_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = 1.0f / sqrtf(v.f);
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(ftanh_s, {
	    GPGPU_INST_BIN(INST_FTANH_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = tanhf(v.f);
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(fsigmoid_s, {
	    GPGPU_INST_BIN(INST_FSIGMOID_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = 1.0f / (1.0f + expf(-v.f));
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(fsin_s, {
	    GPGPU_INST_BIN(INST_FSIN_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = sinf(v.f);
	    F(rd) = v.u;
	})

EXEC_FUNC_FP(fcos_s, {
	    GPGPU_INST_BIN(INST_FCOS_S, (uint32_t)rd, src1);
	    union { uint32_t u; float f; } v;
	    v.u = src1;
	    v.f = cosf(v.f);
	    F(rd) = v.u;
	})

/* ========== SPECIAL ============ */



/* ============================================= Instruction Table =================================================== */
typedef void (*exec_func_t)(exec_ctx_t *ctx, int lane_id);
typedef struct opcode_entry {
	    uint32_t mask;
	    uint32_t match;
	    exec_func_t exec;
	    int type;
} opcode_entry_t;

#define INSTRUCTION_LIST \
	    /* RV32IM */ \
	    X(jal,          "??????? ????? ????? ??? ????? 11011 11", TYPE_J); \
	    X(jalr,         "??????? ????? ????? 000 ????? 11001 11", TYPE_I); \
	    X(beq,          "??????? ????? ????? 000 ????? 11000 11", TYPE_B); \
	    X(bne,          "??????? ????? ????? 001 ????? 11000 11", TYPE_B); \
	    X(blt,          "??????? ????? ????? 100 ????? 11000 11", TYPE_B); \
	    X(bge,          "??????? ????? ????? 101 ????? 11000 11", TYPE_B); \
	    X(bltu,         "??????? ????? ????? 110 ????? 11000 11", TYPE_B); \
	    X(bgeu,         "??????? ????? ????? 111 ????? 11000 11", TYPE_B); \
	    X(lb,           "??????? ????? ????? 000 ????? 00000 11", TYPE_I); \
	    X(lh,           "??????? ????? ????? 001 ????? 00000 11", TYPE_I); \
	    X(lw,           "??????? ????? ????? 010 ????? 00000 11", TYPE_I); \
	    X(lbu,          "??????? ????? ????? 100 ????? 00000 11", TYPE_I); \
	    X(lhu,          "??????? ????? ????? 101 ????? 00000 11", TYPE_I); \
	    X(sb,           "??????? ????? ????? 000 ????? 01000 11", TYPE_S); \
	    X(sh,           "??????? ????? ????? 001 ????? 01000 11", TYPE_S); \
	    X(sw,           "??????? ????? ????? 010 ????? 01000 11", TYPE_S); \
	    X(lui,          "??????? ????? ????? ??? ????? 01101 11", TYPE_U); \
	    X(auipc,        "??????? ????? ????? ??? ????? 00101 11", TYPE_U); \
	    X(addi,         "??????? ????? ????? 000 ????? 00100 11", TYPE_I); \
	    X(slti,         "??????? ????? ????? 010 ????? 00100 11", TYPE_I); \
	    X(sltiu,        "??????? ????? ????? 011 ????? 00100 11", TYPE_I); \
	    X(xori,         "??????? ????? ????? 100 ????? 00100 11", TYPE_I); \
	    X(ori,          "??????? ????? ????? 110 ????? 00100 11", TYPE_I); \
	    X(andi,         "??????? ????? ????? 111 ????? 00100 11", TYPE_I); \
	    X(slli,         "0000000 ????? ????? 001 ????? 00100 11", TYPE_I); \
	    X(srli,         "0000000 ????? ????? 101 ????? 00100 11", TYPE_I); \
	    X(srai,         "0100000 ????? ????? 101 ????? 00100 11", TYPE_I); \
	    X(add,          "0000000 ????? ????? 000 ????? 01100 11", TYPE_R); \
	    X(sub,          "0100000 ????? ????? 000 ????? 01100 11", TYPE_R); \
	    X(sll,          "0000000 ????? ????? 001 ????? 01100 11", TYPE_R); \
	    X(slt,          "0000000 ????? ????? 010 ????? 01100 11", TYPE_R); \
	    X(sltu,         "0000000 ????? ????? 011 ????? 01100 11", TYPE_R); \
	    X(xor,          "0000000 ????? ????? 100 ????? 01100 11", TYPE_R); \
	    X(srl,          "0000000 ????? ????? 101 ????? 01100 11", TYPE_R); \
	    X(sra,          "0100000 ????? ????? 101 ????? 01100 11", TYPE_R); \
	    X(or,           "0000000 ????? ????? 110 ????? 01100 11", TYPE_R); \
	    X(and,          "0000000 ????? ????? 111 ????? 01100 11", TYPE_R); \
	    X(mul,          "0000001 ????? ????? 000 ????? 01100 11", TYPE_R); \
	    X(mulh,         "0000001 ????? ????? 001 ????? 01100 11", TYPE_R); \
	    X(mulhsu,       "0000001 ????? ????? 010 ????? 01100 11", TYPE_R); \
	    X(mulhu,        "0000001 ????? ????? 011 ????? 01100 11", TYPE_R); \
	    X(div,          "0000001 ????? ????? 100 ????? 01100 11", TYPE_R); \
	    X(divu,         "0000001 ????? ????? 101 ????? 01100 11", TYPE_R); \
	    X(rem,          "0000001 ????? ????? 110 ????? 01100 11", TYPE_R); \
	    X(remu,         "0000001 ????? ????? 111 ????? 01100 11", TYPE_R); \
	    X(csrrw,        "??????? ????? ????? 001 ????? 11100 11", TYPE_CSR); \
	    X(csrrs,        "??????? ????? ????? 010 ????? 11100 11", TYPE_CSR); \
	    X(csrrc,        "??????? ????? ????? 011 ????? 11100 11", TYPE_CSR); \
	    X(csrrwi,       "??????? ????? ????? 101 ????? 11100 11", TYPE_CSR); \
	    X(csrrsi,       "??????? ????? ????? 110 ????? 11100 11", TYPE_CSR); \
	    X(csrrci,       "??????? ????? ????? 111 ????? 11100 11", TYPE_CSR); \
	    X(ebreak,       "0000000 00001 00000 000 00000 11100 11", TYPE_I); \
	    /* RV32F */ \
	    X(flw,          "??????? ????? ????? 010 ????? 00001 11", TYPE_I); \
	    X(fsw,          "??????? ????? ????? 010 ????? 01001 11", TYPE_S); \
	    X(fmadd_s,      "?????00 ????? ????? ??? ????? 10000 11", TYPE_F4); \
	    X(fmsub_s,      "?????00 ????? ????? ??? ????? 10001 11", TYPE_F4); \
	    X(fnmsub_s,     "?????00 ????? ????? ??? ????? 10010 11", TYPE_F4); \
	    X(fnmadd_s,     "?????00 ????? ????? ??? ????? 10011 11", TYPE_F4); \
	    X(fadd_s,       "0000000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fsub_s,       "0000100 ????? ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fmul_s,       "0001000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fdiv_s,       "0001100 ????? ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fsqrt_s,      "0101100 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fsgnj_s,      "0010000 ????? ????? 000 ????? 10100 11", TYPE_FR); \
	    X(fsgnjn_s,     "0010000 ????? ????? 001 ????? 10100 11", TYPE_FR); \
	    X(fsgnjx_s,     "0010000 ????? ????? 010 ????? 10100 11", TYPE_FR); \
	    X(fmin_s,       "0010100 ????? ????? 000 ????? 10100 11", TYPE_FR); \
	    X(fmax_s,       "0010100 ????? ????? 001 ????? 10100 11", TYPE_FR); \
	    X(fcvt_w_s,     "1100000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_wu_s,    "1100000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fmv_x_w,      "1110000 00000 ????? 000 ????? 10100 11", TYPE_FR); \
	    X(feq_s,        "1010000 ????? ????? 010 ????? 10100 11", TYPE_FR); \
	    X(flt_s,        "1010000 ????? ????? 001 ????? 10100 11", TYPE_FR); \
	    X(fle_s,        "1010000 ????? ????? 000 ????? 10100 11", TYPE_FR); \
	    X(fclass_s,     "1110000 00000 ????? 001 ????? 10100 11", TYPE_FR); \
	    X(fcvt_s_w,     "1101000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_s_wu,    "1101000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fmv_w_x,      "1111000 00000 ????? 000 ????? 10100 11", TYPE_FR); \
	    /* LP float inst */ \
	    X(fcvt_s_bf16,  "0100010 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_bf16_s,  "0100010 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_s_e4m3,  "0100100 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_e4m3_s,  "0100100 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_s_e5m2,  "0100100 00010 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_e5m2_s,  "0100100 00011 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_s_e2m1,  "0100110 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcvt_e2m1_s,  "0100110 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    /* Scientific functions */ \
	    X(fexp_s,       "0110000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fln_s,        "0110000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(frcp_s,       "0110000 00010 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(frsqrt_s,     "0110000 00011 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(ftanh_s,      "0110000 00100 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fsigmoid_s,   "0110000 00101 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fsin_s,       "0110000 00110 ????? ??? ????? 10100 11", TYPE_FR); \
	    X(fcos_s,       "0110000 00111 ????? ??? ????? 10100 11", TYPE_FR);

static opcode_entry_t opcode_table[NUM_OF_INST];
static size_t opcode_table_count = 0;

static void __attribute__((constructor)) init_opcode_table(void)
{
	    int idx = 0;

	#define X(name, pattern, op_type) \
	    do { \
	        opcode_table[idx].mask = pattern_to_mask(pattern); \
	        opcode_table[idx].match = pattern_to_match(pattern); \
	        opcode_table[idx].exec = exec_##name; \
	        opcode_table[idx].type = op_type; \
	        idx++; \
	    } while(0)

	    INSTRUCTION_LIST

	#undef X

	    opcode_table_count = idx;
	}

static opcode_entry_t *lookup_opcode(uint32_t inst)
{
	    for (size_t i = 0; i < opcode_table_count; i++) {
	        if ((inst & opcode_table[i].mask) == opcode_table[i].match) {
	            return &opcode_table[i];
	        }
	    }
	    return NULL;
	}

static int exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{
	    const opcode_entry_t *entry = lookup_opcode(inst);
	    if (!entry) {
	        VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_ILLEGAL_INST, inst);
	        return -1;
	    }

	    exec_ctx_t ctx = {
	        .s = s,
	        .warp = warp,
	        .type = entry->type,
	    };

	    get_warp_ctx(&ctx, inst, entry->type);

	    for (int lane = 0; lane < GPGPU_WARP_SIZE; lane++) {
	        if (warp->active_mask & (1 << lane)) {
	            // 更新 s->simt 中的线程 ID 和 block ID
	            s->simt.thread_id[0] = warp->thread_id_base + lane;
	            s->simt.thread_id[1] = 0;
	            s->simt.thread_id[2] = 0;
	            s->simt.block_id[0] = warp->block_id[0];
	            s->simt.block_id[1] = warp->block_id[1];
	            s->simt.block_id[2] = warp->block_id[2];

	            // 检查是否是 ebreak 指令
	            if (entry->match == MATCH_EBREAK) {
	                warp->active_mask &= ~(1 << lane);
	                continue;
	            }

	            entry->exec(&ctx, lane);
	            warp->lanes[lane].gpr[0].u32 = 0;

	            // 检查是否是 ret 指令（jalr x0, 0(x1)）
	            if (entry->type == TYPE_I && (inst & 0x7f) == 0x67 && ctx.rd == 0 && ctx.rs1 == 1 && ctx.imm == 0) {
	                if (warp->lanes[lane].gpr[1].u32 == 0) {
	                    warp->active_mask &= ~(1 << lane);
	                }
	            }
	        }
	    }

	    return 0;
	}

/* Initialize the warp */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
		                          uint32_t thread_id_base, const uint32_t block_id[3],
		                          uint32_t num_threads,
		                          uint32_t warp_id, uint32_t block_id_linear)
{
	    memset(warp, 0, sizeof(*warp));

	    warp->thread_id_base = thread_id_base;
	    warp->warp_id = warp_id;
	    warp->block_id[0] = block_id[0];
	    warp->block_id[1] = block_id[1];
	    warp->block_id[2] = block_id[2];

	    /* Set active mask */
	    if (num_threads >= GPGPU_WARP_SIZE) {
	        warp->active_mask = 0xFFFFFFFF;
	    } else {
	        warp->active_mask = (1U << num_threads) - 1;
	    }

	    /* Initialize each lane */
	    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
	        GPGPULane *lane = &warp->lanes[i];
	        lane->pc = pc;
	        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
	        lane->active = (warp->active_mask & (1 << i)) != 0;
	        lane->gpr[0].u32 = 0;
	        lane->fpr[0].u32 = 0;
	    }
	}

/* warp execution */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
	    uint32_t cycles = 0;
	    uint32_t last_pc = 0;

	    while (cycles < max_cycles) {
	        if (warp->active_mask == 0) {
	            return 0;
	        }

	        uint32_t pc = warp->lanes[0].pc;
	        if (pc >= s->vram_size) {
	            VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_VRAM_FAULT, pc);
	            return -1;
	        }

	        if (cycles > 0 && pc == last_pc) {
	            return 0;
	        }
	        last_pc = pc;

	        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
	        int ret = exec_one_inst(s, warp, inst);

	        if (ret == 1) {
	            return 0;
	        } else if (ret == -1) {
	            return -1;
	        }

	        cycles++;
	    }

	    VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_WARP_TIMEOUT, max_cycles);
	    s->error_status |= GPGPU_ERR_KERNEL_FAULT;
	    return -1;
	}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
	    uint32_t grid_dim[3] = {
	        s->kernel.grid_dim[0],
	        s->kernel.grid_dim[1],
	        s->kernel.grid_dim[2]
	    };
	    uint32_t block_dim[3] = {
	        s->kernel.block_dim[0],
	        s->kernel.block_dim[1],
	        s->kernel.block_dim[2]
	    };

	    uint32_t kernel_addr = s->kernel.kernel_addr;
	    uint32_t threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];

	    for (uint32_t z = 0; z < grid_dim[2]; z++) {
	        for (uint32_t y = 0; y < grid_dim[1]; y++) {
	            for (uint32_t x = 0; x < grid_dim[0]; x++) {
	                uint32_t block_id[3] = {x, y, z};
	                uint32_t block_id_linear = z * grid_dim[0] * grid_dim[1] + y * grid_dim[0] + x;

	                uint32_t num_warps = (threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;

	                for (uint32_t warp_id = 0; warp_id < num_warps; warp_id++) {
	                    GPGPUWarp warp;
	                    uint32_t thread_id_base = warp_id * GPGPU_WARP_SIZE;
	                    uint32_t num_threads = threads_per_block - thread_id_base;
	                    if (num_threads > GPGPU_WARP_SIZE) {
	                        num_threads = GPGPU_WARP_SIZE;
	                    }

	                    gpgpu_core_init_warp(&warp, kernel_addr, thread_id_base,
	                                        block_id, num_threads,
	                                        warp_id, block_id_linear);

	                    int ret = gpgpu_core_exec_warp(s, &warp, 100000);
	                    if (ret != 0) {
	                        VPU_EVENT(event_ring, EVENT_ERROR_EVENT, VPU_ERR_WARP_FAILED, ret);
	                        return -1;
	                    }
	                }
	            }
	        }
	    }
	    return 0;
	}