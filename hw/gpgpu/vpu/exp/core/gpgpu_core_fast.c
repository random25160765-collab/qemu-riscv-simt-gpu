/*
 * gpgpu_core_fast.c — 高性能 RISC-V SIMT 解释器
 *
 * 优化策略：
 *   1. 压缩前缀树解码 (O(log N) bit-checks vs O(N) 线性扫描)
 *   2. 跳转表 + computed goto (消除函数调用开销)
 *   3. 线程化解释器 (每条指令直接跳转下一条)
 *
 * 与原始 gpgpu_core.c 共用 inst.h 中的 EXEC_FUNC 宏。
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "state.h"
#include "gpgpu_core.h"
#include "lpfp.h"
#include "memory.h"
#include "inst.h"
#include "utils.h"
#include "proto.h"
#include "decode_trie.h"

/* ============================================================
 * 类型定义 (与 gpgpu_core.c 同步)
 * ============================================================ */
typedef enum {
    TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR,
    TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4,
} inst_type_t;

typedef struct exec_ctx {
    GPGPUState *s;
    GPGPUWarp *warp;
    int rd; int rs1; int rs2; int rs3;
    int32_t imm;
    int type;
} exec_ctx_t;

/* ============================================================
 * CSR helper
 * ============================================================ */
static uint32_t csr_read(GPGPULane *l, uint16_t csr_addr) {
    switch (csr_addr) {
        case CSR_MHARTID: return l->mhartid;
        case CSR_FFLAGS:  return l->fcsr & 0x1F;
        case CSR_FRM:     return (l->fcsr >> 5) & 0x7;
        case CSR_FCSR:    return l->fcsr;
        default: return 0;
    }
}
static void csr_write(GPGPULane *l, uint16_t csr_addr, uint32_t val) {
    switch (csr_addr) {
        case CSR_FFLAGS: l->fcsr = (l->fcsr & ~0x1F) | (val & 0x1F); break;
        case CSR_FRM:    l->fcsr = (l->fcsr & ~0xE0) | ((val & 0x7) << 5); break;
        case CSR_FCSR:   l->fcsr = val; break;
        default: break;
    }
}

#include "lpfp.h"
#include <math.h>


/* ============================================================
 * 指令表 — 用于构建 trie
 * ============================================================ */
/* opcode_entry_t defined in decode_trie.h */

#define INSTRUCTION_LIST \
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
    X(fcvt_s_bf16,  "0100010 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_bf16_s,  "0100010 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e4m3,  "0100100 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e4m3_s,  "0100100 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e5m2,  "0100100 00010 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e5m2_s,  "0100100 00011 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e2m1,  "0100110 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e2m1_s,  "0100110 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fexp_s,       "0110000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fln_s,        "0110000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(frcp_s,       "0110000 00010 ????? ??? ????? 10100 11", TYPE_FR); \
    X(frsqrt_s,     "0110000 00011 ????? ??? ????? 10100 11", TYPE_FR); \
    X(ftanh_s,      "0110000 00100 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsigmoid_s,   "0110000 00101 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsin_s,       "0110000 00110 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcos_s,       "0110000 00111 ????? ??? ????? 10100 11", TYPE_FR);

#define NUM_OF_INST 300

/* ============================================================
 * 全局状态
 * ============================================================ */
static opcode_entry_t opcode_table[NUM_OF_INST];
static size_t opcode_table_count = 0;
static DecodeTrie decode_trie;

static void __attribute__((constructor)) init_fast_decoder(void)
{
    int idx = 0;
    #define X(name, pattern, op_type) \
        do { \
            opcode_table[idx].mask  = pattern_to_mask(pattern); \
            opcode_table[idx].match = pattern_to_match(pattern); \
            opcode_table[idx].exec  = NULL; /* computed goto, no func ptr needed */ \
            opcode_table[idx].type  = op_type; \
            idx++; \
        } while(0)
    INSTRUCTION_LIST
    #undef X
    opcode_table_count = idx;

    /* 构建压缩前缀树 */
    decode_trie_build(&decode_trie, opcode_table, opcode_table_count);
}

/* ============================================================
 * 快速查找 — inline
 * ============================================================ */
static inline uint16_t fast_decode(uint32_t inst)
{
    uint16_t idx = decode_trie.root_idx;
    const TrieNode *nodes = decode_trie.nodes;
    while (nodes[idx].bit_pos >= 0) {
        int bit = (inst >> nodes[idx].bit_pos) & 1;
        uint16_t next = nodes[idx].child[bit];
        if (next == 0) return 0;
        idx = next;
    }
    return nodes[idx].instr_id;
}

/* ============================================================
 * 线程化解释器 — computed goto
 * ============================================================ */

/* Warp 初始化 — 与原始版本相同 */
void gpgpu_core_fast_init_warp(GPGPUWarp *warp, uint32_t pc,
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
        lane->fp_status.float_exception_flags |= float_flag_inexact;
        lane->fp_status.float_rounding_mode = float_round_nearest_even;
        lane->fp_status.default_nan_pattern = 0x7f;  /* RISC-V canonical NaN */
    }
}

/* 线程化解释器主循环 */
int gpgpu_core_fast_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    GPGPULane *l = &warp->lanes[0];  /* 当前活动 lane */
    exec_ctx_t ctx_val = { .s = s, .warp = warp };
    exec_ctx_t *ctx = &ctx_val;
    uint32_t inst, pc;

    /* 跳转表：指令索引 → label 地址 */
    static void *dispatch[NUM_OF_INST];
    static int dispatch_ready = 0;

    if (!dispatch_ready) {
        /* 填充跳转表（与 INSTRUCTION_LIST 同序） */
        size_t di = 0;
        #define X(name, p, t) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
        #undef X
        dispatch_ready = 1;
    }

    /* 设置 SIMT 上下文（供 CTRL 寄存器读取） */
    s->simt.thread_id[0] = warp->thread_id_base;
    s->simt.thread_id[1] = 0;
    s->simt.thread_id[2] = 0;
    s->simt.block_id[0] = warp->block_id[0];
    s->simt.block_id[1] = warp->block_id[1];
    s->simt.block_id[2] = warp->block_id[2];

    /* 入口 */
    goto fetch;

    /* ============================================================
     * 指令实现 — computed goto 线程化
     * 每条指令结束后通过 NEXT() 直接跳转到下一条的 label
     * ============================================================ */
    #define NEXT() do { \
        if (++cycles >= max_cycles) goto timeout; \
        l->gpr[0].u32 = 0; \
        pc = l->pc; \
        if (pc >= s->vram_size) goto vram_fault; \
        inst = *(uint32_t *)(s->vram_ptr + pc); \
        uint16_t id = fast_decode(inst); \
        if (id == 0) goto illegal; \
        ctx->rd  = BITS(inst, 11, 7); \
        ctx->rs1 = BITS(inst, 19, 15); \
        ctx->rs2 = BITS(inst, 24, 20); \
        ctx->rs3 = BITS(inst, 31, 27); \
        goto *dispatch[id - 1]; \
    } while(0)

fetch:
    l->gpr[0].u32 = 0;
    pc = l->pc;
    if (pc >= s->vram_size) goto vram_fault;
    inst = *(uint32_t *)(s->vram_ptr + pc);
    {
        uint16_t id = fast_decode(inst);
        if (id == 0) goto illegal;
        ctx->rd  = BITS(inst, 11, 7);
        ctx->rs1 = BITS(inst, 19, 15);
        ctx->rs2 = BITS(inst, 24, 20);
        ctx->rs3 = BITS(inst, 31, 27);
        goto *dispatch[id - 1];
    }

    /* —— 以下是所有指令的 computed goto label —— */

op_jal: {
    ctx->imm = immJ(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->pc + 4;
    l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_jalr: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32;
    l->gpr[ctx->rd].u32 = l->pc + 4;
    l->pc = (src1 + ctx->imm) & ~1;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_beq: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if (src1 == src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_bne: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if (src1 != src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_blt: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if ((int32_t)src1 < (int32_t)src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_bge: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if ((int32_t)src1 >= (int32_t)src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_bltu: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if (src1 < src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_bgeu: {
    ctx->imm = immB(inst);
    uint32_t old_pc = l->pc;
    uint32_t src1 = l->gpr[ctx->rs1].u32, src2 = l->gpr[ctx->rs2].u32;
    if (src1 >= src2) l->pc += ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lb: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    uint32_t addr = l->gpr[ctx->rs1].u32 + ctx->imm;
    uint8_t v = gpu_read(ctx->s, addr, 1);
    l->gpr[ctx->rd].i32 = (int32_t)(v << 24) >> 24;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lh: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    uint32_t addr = l->gpr[ctx->rs1].u32 + ctx->imm;
    uint16_t v = gpu_read(ctx->s, addr, 2);
    l->gpr[ctx->rd].i32 = (int32_t)(v << 16) >> 16;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lw: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    uint32_t addr = l->gpr[ctx->rs1].u32 + ctx->imm;
    l->gpr[ctx->rd].u32 = gpu_read(ctx->s, addr, 4);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lbu: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = gpu_read(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 1);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lhu: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = gpu_read(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 2);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sb: {
    ctx->imm = immS(inst);
    uint32_t old_pc = l->pc;
    gpu_write(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 1, l->gpr[ctx->rs2].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sh: {
    ctx->imm = immS(inst);
    uint32_t old_pc = l->pc;
    gpu_write(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 2, l->gpr[ctx->rs2].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sw: {
    ctx->imm = immS(inst);
    uint32_t old_pc = l->pc;
    gpu_write(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 4, l->gpr[ctx->rs2].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_lui: {
    ctx->imm = immU(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_auipc: {
    ctx->imm = immU(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->pc + ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_addi: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 + ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_slti: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = ((int32_t)l->gpr[ctx->rs1].u32 < ctx->imm) ? 1 : 0;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sltiu: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = (l->gpr[ctx->rs1].u32 < (uint32_t)ctx->imm) ? 1 : 0;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_xori: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 ^ ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_ori: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 | ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_andi: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 & ctx->imm;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_slli: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 << (ctx->imm & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_srli: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 >> (ctx->imm & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_srai: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->gpr[ctx->rd].i32 = (int32_t)l->gpr[ctx->rs1].u32 >> (ctx->imm & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_add: {
    uint32_t old_pc = l->pc;
    ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 + l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sub: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 - l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sll: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 << (l->gpr[ctx->rs2].u32 & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_slt: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = ((int32_t)l->gpr[ctx->rs1].u32 < (int32_t)l->gpr[ctx->rs2].u32) ? 1 : 0;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sltu: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = (l->gpr[ctx->rs1].u32 < l->gpr[ctx->rs2].u32) ? 1 : 0;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_xor: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 ^ l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_srl: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 >> (l->gpr[ctx->rs2].u32 & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_sra: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].i32 = (int32_t)l->gpr[ctx->rs1].u32 >> (l->gpr[ctx->rs2].u32 & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_or: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 | l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

op_and: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 & l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

/* RV32M */
op_mul: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32 * l->gpr[ctx->rs2].u32;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_mulh: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = (uint32_t)(((int64_t)(int32_t)l->gpr[ctx->rs1].u32 * (int64_t)(int32_t)l->gpr[ctx->rs2].u32) >> 32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_mulhsu: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = (uint32_t)(((int64_t)(int32_t)l->gpr[ctx->rs1].u32 * (uint64_t)l->gpr[ctx->rs2].u32) >> 32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_mulhu: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->gpr[ctx->rd].u32 = (uint32_t)(((uint64_t)l->gpr[ctx->rs1].u32 * (uint64_t)l->gpr[ctx->rs2].u32) >> 32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_div: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    int32_t s1 = l->gpr[ctx->rs1].i32, s2 = l->gpr[ctx->rs2].i32;
    l->gpr[ctx->rd].i32 = (s2 == 0) ? -1 : s1 / s2;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_divu: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    uint32_t s1 = l->gpr[ctx->rs1].u32, s2 = l->gpr[ctx->rs2].u32;
    l->gpr[ctx->rd].u32 = (s2 == 0) ? 0xFFFFFFFF : s1 / s2;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_rem: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    int32_t s1 = l->gpr[ctx->rs1].i32, s2 = l->gpr[ctx->rs2].i32;
    l->gpr[ctx->rd].i32 = (s2 == 0) ? s1 : (s1 % s2);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_remu: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    uint32_t s1 = l->gpr[ctx->rs1].u32, s2 = l->gpr[ctx->rs2].u32;
    l->gpr[ctx->rd].u32 = (s2 == 0) ? s1 : s1 % s2;
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

/* ebreak */
op_ebreak: {
    warp->active_mask &= ~1;  /* single-lane: deactivate lane 0 */
    return 0;
}

/* CSR */
op_csrrw: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    csr_write(l, c, l->gpr[ctx->rs1].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_csrrs: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    if (ctx->rs1 != 0) csr_write(l, c, o | l->gpr[ctx->rs1].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_csrrc: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    if (ctx->rs1 != 0) csr_write(l, c, o & ~l->gpr[ctx->rs1].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_csrrwi: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    csr_write(l, c, (uint32_t)ctx->rs1);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_csrrsi: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    if (ctx->rs1 != 0) csr_write(l, c, o | (uint32_t)ctx->rs1);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_csrrci: {
    ctx->imm = immCSR(inst);
    uint32_t old_pc = l->pc;
    uint16_t c = (uint16_t)ctx->imm;
    uint32_t o = csr_read(l, c);
    l->gpr[ctx->rd].u32 = o;
    if (ctx->rs1 != 0) csr_write(l, c, o & ~(uint32_t)ctx->rs1);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

/* RV32F — 简化的 computed goto 版本 */
op_fadd_s: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->fp_status.float_exception_flags = float_flag_inexact;
    l->fp_status.float_rounding_mode = (l->fcsr >> 5) & 0x7;
    if (l->fp_status.float_rounding_mode > 4) l->fp_status.float_rounding_mode = 0;
    l->fpr[ctx->rd].u32 = float32_add(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    l->fcsr = (l->fcsr & ~0x1F) | (l->fp_status.float_exception_flags & 0x1F);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_fsub_s: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->fp_status.float_exception_flags = float_flag_inexact;
    l->fpr[ctx->rd].u32 = float32_sub(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_fmul_s: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->fp_status.float_exception_flags = float_flag_inexact;
    l->fpr[ctx->rd].u32 = float32_mul(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_fdiv_s: {
    uint32_t old_pc = l->pc; ctx->imm = 0;
    l->fp_status.float_exception_flags = float_flag_inexact;
    l->fpr[ctx->rd].u32 = float32_div(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

/* ============================================================
 * FP 指令统一 inline — computed goto 跳转表
 *
 * FP_SETUP: 同步 fcsr→fp_status，设置 rounding mode 和 inexact flag
 * FP_TEARDOWN: 同步 fp_status→fcsr，更新 PC
 * ============================================================ */
#define FP_SETUP() do { \
    uint32_t _old_pc = l->pc; \
    ctx->imm = 0; \
    uint8_t _frm = (l->fcsr >> 5) & 0x7; \
    l->fp_status.float_rounding_mode = (_frm <= 4) ? _frm : 0; \
    l->fp_status.float_exception_flags = float_flag_inexact; \
    (void)_old_pc

#define FP_DONE() \
    l->fcsr = (l->fcsr & ~0x1F) | (l->fp_status.float_exception_flags & 0x1F); \
    if (l->pc == _old_pc) l->pc += 4; \
} while(0)

/* ---- fsqrt ---- */
op_fsqrt_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_sqrt(l->fpr[ctx->rs1].u32, &l->fp_status);
    FP_DONE(); NEXT();
}

/* ---- FMA x4 ---- */
op_fmadd_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_muladd(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, l->fpr[ctx->rs3].u32, 0, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fmsub_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_muladd(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, l->fpr[ctx->rs3].u32, float_muladd_negate_c, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fnmsub_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_muladd(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, l->fpr[ctx->rs3].u32, float_muladd_negate_product, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fnmadd_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_muladd(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, l->fpr[ctx->rs3].u32, float_muladd_negate_result, &l->fp_status);
    FP_DONE(); NEXT();
}

/* ---- 符号注入 ---- */
op_fsgnj_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = (l->fpr[ctx->rs1].u32 & ~0x80000000) | (l->fpr[ctx->rs2].u32 & 0x80000000);
    FP_DONE(); NEXT();
}
op_fsgnjn_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = (l->fpr[ctx->rs1].u32 & ~0x80000000) | ((~l->fpr[ctx->rs2].u32) & 0x80000000);
    FP_DONE(); NEXT();
}
op_fsgnjx_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = l->fpr[ctx->rs1].u32 ^ (l->fpr[ctx->rs2].u32 & 0x80000000);
    FP_DONE(); NEXT();
}

/* ---- 最值 ---- */
op_fmin_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_min(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fmax_s: { FP_SETUP();
    l->fpr[ctx->rd].u32 = float32_max(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    FP_DONE(); NEXT();
}

/* ---- 比较 (整数结果写入 gpr) ---- */
op_feq_s: { FP_SETUP();
    l->gpr[ctx->rd].u32 = float32_eq(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    FP_DONE(); NEXT();
}
op_flt_s: { FP_SETUP();
    l->gpr[ctx->rd].u32 = float32_lt(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fle_s: { FP_SETUP();
    l->gpr[ctx->rd].u32 = float32_le(l->fpr[ctx->rs1].u32, l->fpr[ctx->rs2].u32, &l->fp_status);
    FP_DONE(); NEXT();
}

/* ---- 转换 fcvt.w.s / fcvt.wu.s ---- */
op_fcvt_w_s: { FP_SETUP();
    float32 _f = l->fpr[ctx->rs1].u32;
    if (float32_is_quiet_nan(_f, &l->fp_status) || float32_is_signaling_nan(_f, &l->fp_status))
        l->gpr[ctx->rd].i32 = 0x7FFFFFFF;
    else {
        float32 _mx = int32_to_float32(0x7FFFFFFF, &l->fp_status);
        float32 _mn = int32_to_float32(0x80000000, &l->fp_status);
        if (float32_le(_mx, _f, &l->fp_status)) l->gpr[ctx->rd].i32 = 0x7FFFFFFF;
        else if (float32_lt(_f, _mn, &l->fp_status)) l->gpr[ctx->rd].i32 = 0x80000000;
        else l->gpr[ctx->rd].i32 = float32_to_int32(_f, &l->fp_status);
    }
    FP_DONE(); NEXT();
}
op_fcvt_wu_s: { FP_SETUP();
    float32 _f = l->fpr[ctx->rs1].u32;
    if (float32_is_quiet_nan(_f, &l->fp_status) || float32_is_signaling_nan(_f, &l->fp_status))
        l->gpr[ctx->rd].u32 = 0xFFFFFFFF;
    else if (float32_lt(_f, 0, &l->fp_status)) l->gpr[ctx->rd].u32 = 0;
    else {
        float32 _mu = uint32_to_float32(0xFFFFFFFF, &l->fp_status);
        if (float32_le(_mu, _f, &l->fp_status)) l->gpr[ctx->rd].u32 = 0xFFFFFFFF;
        else l->gpr[ctx->rd].u32 = float32_to_uint32(_f, &l->fp_status);
    }
    FP_DONE(); NEXT();
}

/* ---- 转换 fcvt.s.w / fcvt.s.wu ---- */
op_fcvt_s_w: { FP_SETUP();
    l->fpr[ctx->rd].u32 = int32_to_float32(l->gpr[ctx->rs1].i32, &l->fp_status);
    FP_DONE(); NEXT();
}
op_fcvt_s_wu: { FP_SETUP();
    l->fpr[ctx->rd].u32 = uint32_to_float32(l->gpr[ctx->rs1].u32, &l->fp_status);
    FP_DONE(); NEXT();
}

/* ---- 数据移动 ---- */
op_fmv_w_x: { FP_SETUP();
    l->fpr[ctx->rd].u32 = l->gpr[ctx->rs1].u32;
    FP_DONE(); NEXT();
}
op_fmv_x_w: { FP_SETUP();
    l->gpr[ctx->rd].u32 = l->fpr[ctx->rs1].u32;
    FP_DONE(); NEXT();
}

/* ---- 分类 ---- */
op_fclass_s: { FP_SETUP();
    uint32_t _b = l->fpr[ctx->rs1].u32;
    uint32_t _e = (_b >> 23) & 0xFF, _m = _b & 0x7FFFFF, _s = (_b >> 31) & 1;
    int _r = 0;
    if (_e == 0xFF) { if (_m == 0) _r = _s ? (1<<0) : (1<<7); else _r = _s ? (1<<9) : (1<<8); }
    else if (_e == 0) { if (_m == 0) _r = _s ? (1<<3) : (1<<4); else _r = _s ? (1<<2) : (1<<5); }
    else { _r = _s ? (1<<1) : (1<<6); }
    l->gpr[ctx->rd].u32 = _r;
    FP_DONE(); NEXT();
}

/* ---- flw / fsw (需要 imm) ---- */
op_flw: {
    ctx->imm = immI(inst);
    uint32_t old_pc = l->pc;
    l->fpr[ctx->rd].u32 = gpu_read(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 4);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}
op_fsw: {
    ctx->imm = immS(inst);
    uint32_t old_pc = l->pc;
    gpu_write(ctx->s, l->gpr[ctx->rs1].u32 + ctx->imm, 4, l->fpr[ctx->rs2].u32);
    if (l->pc == old_pc) l->pc += 4;
    NEXT();
}

/* ---- LP 浮点转换 ---- */
op_fcvt_s_bf16: { FP_SETUP();
    l->fpr[ctx->rd].u32 = bf16_to_f32(l->fpr[ctx->rs1].bf16);
    FP_DONE(); NEXT();
}
op_fcvt_bf16_s: { FP_SETUP();
    l->fpr[ctx->rd].bf16 = f32_to_bf16(l->fpr[ctx->rs1].u32);
    FP_DONE(); NEXT();
}
op_fcvt_s_e4m3: { FP_SETUP();
    l->fpr[ctx->rd].u32 = e4m3_to_f32(l->fpr[ctx->rs1].e4m3);
    FP_DONE(); NEXT();
}
op_fcvt_e4m3_s: { FP_SETUP();
    l->fpr[ctx->rd].e4m3 = f32_to_e4m3(l->fpr[ctx->rs1].u32);
    FP_DONE(); NEXT();
}
op_fcvt_s_e5m2: { FP_SETUP();
    l->fpr[ctx->rd].u32 = e5m2_to_f32(l->fpr[ctx->rs1].e5m2);
    FP_DONE(); NEXT();
}
op_fcvt_e5m2_s: { FP_SETUP();
    l->fpr[ctx->rd].e5m2 = f32_to_e5m2(l->fpr[ctx->rs1].u32);
    FP_DONE(); NEXT();
}
op_fcvt_s_e2m1: { FP_SETUP();
    l->fpr[ctx->rd].u32 = e2m1_to_f32(l->fpr[ctx->rs1].e2m1);
    FP_DONE(); NEXT();
}
op_fcvt_e2m1_s: { FP_SETUP();
    l->fpr[ctx->rd].e2m1 = f32_to_e2m1(l->fpr[ctx->rs1].u32);
    FP_DONE(); NEXT();
}

/* ---- 科学计算 ---- */
#define FP_SCALAR(op_name, expr) \
op_##op_name: { FP_SETUP(); { union { uint32_t u; float f; } _v; _v.u = l->fpr[ctx->rs1].u32; _v.f = expr; l->fpr[ctx->rd].u32 = _v.u; } FP_DONE(); NEXT(); }

FP_SCALAR(fexp_s,     expf(_v.f))
FP_SCALAR(fln_s,      logf(_v.f))
FP_SCALAR(frcp_s,     1.0f / _v.f)
FP_SCALAR(frsqrt_s,   1.0f / sqrtf(_v.f))
FP_SCALAR(ftanh_s,    tanhf(_v.f))
FP_SCALAR(fsigmoid_s, 1.0f / (1.0f + expf(-_v.f)))
FP_SCALAR(fsin_s,     sinf(_v.f))
FP_SCALAR(fcos_s,     cosf(_v.f))

#undef FP_SCALAR
#undef FP_SETUP
#undef FP_DONE

    /* 错误出口 */
illegal:
    GPGPU_EVENT(event_ring, EVENT_ERROR_EVENT, 0x03, inst);
    return -1;
vram_fault:
    GPGPU_EVENT(event_ring, EVENT_ERROR_EVENT, 0x01, pc);
    return -1;
timeout:
    s->error_status |= GPGPU_ERR_KERNEL_FAULT;
    return -1;
}

/* Kernel dispatch — 与原始版本相同 */
int gpgpu_core_fast_exec_kernel(GPGPUState *s)
{
    uint32_t grid_dim[3] = { s->kernel.grid_dim[0], s->kernel.grid_dim[1], s->kernel.grid_dim[2] };
    uint32_t block_dim[3] = { s->kernel.block_dim[0], s->kernel.block_dim[1], s->kernel.block_dim[2] };
    uint32_t kernel_addr = s->kernel.kernel_addr;
    uint32_t threads_per_block = block_dim[0] * block_dim[1] * block_dim[2];

    for (uint32_t z = 0; z < grid_dim[2]; z++) {
        for (uint32_t y = 0; y < grid_dim[1]; y++) {
            for (uint32_t x = 0; x < grid_dim[0]; x++) {
                uint32_t block_id[3] = {x, y, z};
                uint32_t blk_linear = z * grid_dim[0] * grid_dim[1] + y * grid_dim[0] + x;
                uint32_t num_warps = (threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;

                for (uint32_t w = 0; w < num_warps; w++) {
                    GPGPUWarp warp;
                    uint32_t tid_base = w * GPGPU_WARP_SIZE;
                    uint32_t n_threads = threads_per_block - tid_base;
                    if (n_threads > GPGPU_WARP_SIZE) n_threads = GPGPU_WARP_SIZE;

                    gpgpu_core_fast_init_warp(&warp, kernel_addr, tid_base,
                                              block_id, n_threads, w, blk_linear);

                    int ret = gpgpu_core_fast_exec_warp(s, &warp, 100000000);
                    if (ret != 0) return -1;
                }
            }
        }
    }
    return 0;
}
