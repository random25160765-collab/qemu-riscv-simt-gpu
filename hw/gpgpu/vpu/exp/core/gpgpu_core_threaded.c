/*
 * gpgpu_core_threaded.c — 线索化解释器（编译时预译码，运行时零开销）
 *
 * 思路：
 *   load 时：遍历 kernel .bin，每条指令通过 trie 查得 handler 标签地址，
 *            存入 void*[] 数组（同时存预解码的操作数字段）。
 *   运行时：goto **ip++  →  执行  →  goto **ip++  →  ...
 *   零取指、零译码、零查表。
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "gpgpu_core.h"
#include "lpfp.h"
#include "memory.h"
#include "inst.h"
#include "utils.h"
#include "proto.h"
#include "decode_trie.h"
#include <math.h>

/* ============================================================
 * 线索化操作：handler + 预解码字段
 * ============================================================ */
typedef struct {
    void    *handler;          /* computed goto 标签地址 */
    uint32_t inst;             /* 原始 32-bit 指令（供分支译码） */
    int32_t  imm;              /* 预解码立即数 */
    int8_t   rd, rs1, rs2, rs3; /* 预解码寄存器号 */
    int32_t  branch_tgt;       /* 分支目标在数组中的索引，-1=顺序 */
} ThOp;

/* ============================================================
 * 跳转表（与 gpgpu_core_fast.c 共用同一套 INSTRUCTION_LIST）
 * ============================================================ */
#define NUM_OF_INST 300
static void *dispatch[NUM_OF_INST];
static int dispatch_ready = 0;

typedef enum {
    TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR,
    TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4,
} inst_type_t;

#define INSTRUCTION_LIST \
    X(jal,      "??????? ????? ????? ??? ????? 11011 11", TYPE_J, immJ); \
    X(jalr,     "??????? ????? ????? 000 ????? 11001 11", TYPE_I, immI); \
    X(beq,      "??????? ????? ????? 000 ????? 11000 11", TYPE_B, immB); \
    X(bne,      "??????? ????? ????? 001 ????? 11000 11", TYPE_B, immB); \
    X(blt,      "??????? ????? ????? 100 ????? 11000 11", TYPE_B, immB); \
    X(bge,      "??????? ????? ????? 101 ????? 11000 11", TYPE_B, immB); \
    X(bltu,     "??????? ????? ????? 110 ????? 11000 11", TYPE_B, immB); \
    X(bgeu,     "??????? ????? ????? 111 ????? 11000 11", TYPE_B, immB); \
    X(lb,       "??????? ????? ????? 000 ????? 00000 11", TYPE_I, immI); \
    X(lh,       "??????? ????? ????? 001 ????? 00000 11", TYPE_I, immI); \
    X(lw,       "??????? ????? ????? 010 ????? 00000 11", TYPE_I, immI); \
    X(lbu,      "??????? ????? ????? 100 ????? 00000 11", TYPE_I, immI); \
    X(lhu,      "??????? ????? ????? 101 ????? 00000 11", TYPE_I, immI); \
    X(sb,       "??????? ????? ????? 000 ????? 01000 11", TYPE_S, immS); \
    X(sh,       "??????? ????? ????? 001 ????? 01000 11", TYPE_S, immS); \
    X(sw,       "??????? ????? ????? 010 ????? 01000 11", TYPE_S, immS); \
    X(lui,      "??????? ????? ????? ??? ????? 01101 11", TYPE_U, immU); \
    X(auipc,    "??????? ????? ????? ??? ????? 00101 11", TYPE_U, immU); \
    X(addi,     "??????? ????? ????? 000 ????? 00100 11", TYPE_I, immI); \
    X(slti,     "??????? ????? ????? 010 ????? 00100 11", TYPE_I, immI); \
    X(sltiu,    "??????? ????? ????? 011 ????? 00100 11", TYPE_I, immI); \
    X(xori,     "??????? ????? ????? 100 ????? 00100 11", TYPE_I, immI); \
    X(ori,      "??????? ????? ????? 110 ????? 00100 11", TYPE_I, immI); \
    X(andi,     "??????? ????? ????? 111 ????? 00100 11", TYPE_I, immI); \
    X(slli,     "0000000 ????? ????? 001 ????? 00100 11", TYPE_I, immI); \
    X(srli,     "0000000 ????? ????? 101 ????? 00100 11", TYPE_I, immI); \
    X(srai,     "0100000 ????? ????? 101 ????? 00100 11", TYPE_I, immI); \
    X(add,      "0000000 ????? ????? 000 ????? 01100 11", TYPE_R, imm0); \
    X(sub,      "0100000 ????? ????? 000 ????? 01100 11", TYPE_R, imm0); \
    X(sll,      "0000000 ????? ????? 001 ????? 01100 11", TYPE_R, imm0); \
    X(slt,      "0000000 ????? ????? 010 ????? 01100 11", TYPE_R, imm0); \
    X(sltu,     "0000000 ????? ????? 011 ????? 01100 11", TYPE_R, imm0); \
    X(xor,      "0000000 ????? ????? 100 ????? 01100 11", TYPE_R, imm0); \
    X(srl,      "0000000 ????? ????? 101 ????? 01100 11", TYPE_R, imm0); \
    X(sra,      "0100000 ????? ????? 101 ????? 01100 11", TYPE_R, imm0); \
    X(or,       "0000000 ????? ????? 110 ????? 01100 11", TYPE_R, imm0); \
    X(and,      "0000000 ????? ????? 111 ????? 01100 11", TYPE_R, imm0); \
    X(mul,      "0000001 ????? ????? 000 ????? 01100 11", TYPE_R, imm0); \
    X(mulh,     "0000001 ????? ????? 001 ????? 01100 11", TYPE_R, imm0); \
    X(mulhsu,   "0000001 ????? ????? 010 ????? 01100 11", TYPE_R, imm0); \
    X(mulhu,    "0000001 ????? ????? 011 ????? 01100 11", TYPE_R, imm0); \
    X(div,      "0000001 ????? ????? 100 ????? 01100 11", TYPE_R, imm0); \
    X(divu,     "0000001 ????? ????? 101 ????? 01100 11", TYPE_R, imm0); \
    X(rem,      "0000001 ????? ????? 110 ????? 01100 11", TYPE_R, imm0); \
    X(remu,     "0000001 ????? ????? 111 ????? 01100 11", TYPE_R, imm0); \
    X(csrrw,    "??????? ????? ????? 001 ????? 11100 11", TYPE_CSR, immCSR); \
    X(csrrs,    "??????? ????? ????? 010 ????? 11100 11", TYPE_CSR, immCSR); \
    X(csrrc,    "??????? ????? ????? 011 ????? 11100 11", TYPE_CSR, immCSR); \
    X(csrrwi,   "??????? ????? ????? 101 ????? 11100 11", TYPE_CSR, immCSR); \
    X(csrrsi,   "??????? ????? ????? 110 ????? 11100 11", TYPE_CSR, immCSR); \
    X(csrrci,   "??????? ????? ????? 111 ????? 11100 11", TYPE_CSR, immCSR); \
    X(ebreak,   "0000000 00001 00000 000 00000 11100 11", TYPE_I, imm0); \
    X(flw,      "??????? ????? ????? 010 ????? 00001 11", TYPE_I, immI); \
    X(fsw,      "??????? ????? ????? 010 ????? 01001 11", TYPE_S, immS); \
    X(fmadd_s,  "?????00 ????? ????? ??? ????? 10000 11", TYPE_F4, imm0); \
    X(fmsub_s,  "?????00 ????? ????? ??? ????? 10001 11", TYPE_F4, imm0); \
    X(fnmsub_s, "?????00 ????? ????? ??? ????? 10010 11", TYPE_F4, imm0); \
    X(fnmadd_s, "?????00 ????? ????? ??? ????? 10011 11", TYPE_F4, imm0); \
    X(fadd_s,   "0000000 ????? ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fsub_s,   "0000100 ????? ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fmul_s,   "0001000 ????? ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fdiv_s,   "0001100 ????? ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fsqrt_s,  "0101100 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fsgnj_s,  "0010000 ????? ????? 000 ????? 10100 11", TYPE_FR, imm0); \
    X(fsgnjn_s, "0010000 ????? ????? 001 ????? 10100 11", TYPE_FR, imm0); \
    X(fsgnjx_s, "0010000 ????? ????? 010 ????? 10100 11", TYPE_FR, imm0); \
    X(fmin_s,   "0010100 ????? ????? 000 ????? 10100 11", TYPE_FR, imm0); \
    X(fmax_s,   "0010100 ????? ????? 001 ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_w_s, "1100000 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_wu_s,"1100000 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fmv_x_w,  "1110000 00000 ????? 000 ????? 10100 11", TYPE_FR, imm0); \
    X(feq_s,    "1010000 ????? ????? 010 ????? 10100 11", TYPE_FR, imm0); \
    X(flt_s,    "1010000 ????? ????? 001 ????? 10100 11", TYPE_FR, imm0); \
    X(fle_s,    "1010000 ????? ????? 000 ????? 10100 11", TYPE_FR, imm0); \
    X(fclass_s, "1110000 00000 ????? 001 ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_w, "1101000 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_wu,"1101000 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fmv_w_x,  "1111000 00000 ????? 000 ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_bf16,"0100010 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_bf16_s,"0100010 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_e4m3,"0100100 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_e4m3_s,"0100100 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_e5m2,"0100100 00010 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_e5m2_s,"0100100 00011 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_s_e2m1,"0100110 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcvt_e2m1_s,"0100110 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fexp_s,   "0110000 00000 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fln_s,    "0110000 00001 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(frcp_s,   "0110000 00010 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(frsqrt_s, "0110000 00011 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(ftanh_s,  "0110000 00100 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fsigmoid_s,"0110000 00101 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fsin_s,   "0110000 00110 ????? ??? ????? 10100 11", TYPE_FR, imm0); \
    X(fcos_s,   "0110000 00111 ????? ??? ????? 10100 11", TYPE_FR, imm0)

/* imm0: R-type 指令 imm=0 */
static inline int32_t imm0(uint32_t i) { (void)i; return 0; }

/* ============================================================
 * 初始化跳转表 + 构建 trie
 * ============================================================ */
static DecodeTrie threaded_trie;

static opcode_entry_t opcode_table[NUM_OF_INST];
static size_t opcode_count = 0;

static void __attribute__((constructor)) init_threaded(void)
{
    /* 构建 opcode_table 和 trie */
    int idx = 0;
    #define X(name, pattern, op_type, imm_fn); \
        opcode_table[idx].mask  = pattern_to_mask(pattern), \
        opcode_table[idx].match = pattern_to_match(pattern), \
        opcode_table[idx].exec  = NULL, \
        opcode_table[idx].type  = op_type, \
        idx++
    { INSTRUCTION_LIST; }
    #undef X
    opcode_count = idx;
    decode_trie_build(&threaded_trie, opcode_table, opcode_count);

}

/* ============================================================
 * 快速 trie 查找 → 跳转表索引
 * ============================================================ */
static inline uint16_t threaded_decode(uint32_t inst)
{
    uint16_t idx = threaded_trie.root_idx;
    const TrieNode *nodes = threaded_trie.nodes;
    while (nodes[idx].bit_pos >= 0) {
        int bit = (inst >> nodes[idx].bit_pos) & 1;
        uint16_t next = nodes[idx].child[bit];
        if (next == 0) return 0;
        idx = next;
    }
    return nodes[idx].instr_id;
}

/* ============================================================
 * 预译码：kernel .bin → ThOp 数组
 * ============================================================ */
ThOp *threaded_predecode(GPGPUState *s, uint32_t kern_addr, size_t size, int *count)
{
    int n = (int)(size / 4);
    ThOp *code = malloc((n + 1) * sizeof(ThOp)); /* +1: sentinel */

    for (int i = 0; i < n; i++) {
        uint32_t pc  = kern_addr + i * 4;
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        uint16_t id   = threaded_decode(inst);

        if (id == 0) { code[i].handler = NULL; code[i].inst = inst; continue; }

        code[i].handler = dispatch[id - 1];
        code[i].inst    = inst;
        code[i].rd      = (int8_t)BITS(inst, 11, 7);
        code[i].rs1     = (int8_t)BITS(inst, 19, 15);
        code[i].rs2     = (int8_t)BITS(inst, 24, 20);
        code[i].rs3     = (int8_t)BITS(inst, 31, 27);
        code[i].branch_tgt = -1;

        /* 计算 imm */
        switch (inst & 0x7F) {
        case 0x37: code[i].imm = immU(inst);  break;  /* LUI */
        case 0x17: code[i].imm = immU(inst);  break;  /* AUIPC */
        case 0x6F: code[i].imm = immJ(inst);            /* JAL */
                    code[i].branch_tgt = (int)((pc + code[i].imm - kern_addr) / 4);
                    break;
        case 0x67: code[i].imm = immI(inst);  break;  /* JALR */
        case 0x63: code[i].imm = immB(inst);            /* Branch */
                    code[i].branch_tgt = (int)((pc + code[i].imm - kern_addr) / 4);
                    break;
        case 0x03: code[i].imm = immI(inst);  break;  /* Load */
        case 0x23: code[i].imm = immS(inst);  break;  /* Store */
        case 0x13: code[i].imm = immI(inst);  break;  /* OP-IMM */
        case 0x33: code[i].imm = 0;           break;  /* OP */
        case 0x73:
            if ((inst & 0xFE00707F) == 0x00100073)     /* EBREAK */
                code[i].imm = 0;
            else
                code[i].imm = immCSR(inst);            /* CSR */
            break;
        case 0x07:                                     /* FLW */
            code[i].imm = immI(inst);  break;
        case 0x27:                                     /* FSW */
            code[i].imm = immS(inst);  break;
        case 0x43: case 0x47: case 0x4B: case 0x4F:    /* FMA */
            code[i].imm = 0; break;
        case 0x53: code[i].imm = 0; break;              /* FP ops */
        default:   code[i].imm = 0; break;
        }
    }

    /* sentinel: ebreak → 返回 */
    code[n].handler = NULL;  /* sentinel, resolved in exec */
    code[n].inst    = 0;

    *count = n;
    return code;
}

/* ============================================================
 * CSR helpers
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

#define FP_SYNC() do { \
    uint8_t _frm = (l->fcsr >> 5) & 0x7; \
    l->fp_status.float_rounding_mode = (_frm <= 4) ? _frm : 0; \
    l->fp_status.float_exception_flags = float_flag_inexact; \
} while(0)
#define FP_BACK() do { \
    l->fcsr = (l->fcsr & ~0x1F) | (l->fp_status.float_exception_flags & 0x1F); \
} while(0)

/* ============================================================
 * 线索化执行引擎：goto **ip++ 零开销循环
 * ============================================================ */
int gpgpu_core_threaded_exec_warp(GPGPUState *s, GPGPUWarp *warp,
                                   uint32_t max_cycles)
{
    GPGPULane *l = &warp->lanes[0];
    uint32_t cycles = 0;
    (void)max_cycles;

    /* 设置 SIMT 上下文 */
    s->simt.thread_id[0] = warp->thread_id_base;
    s->simt.thread_id[1] = 0; s->simt.thread_id[2] = 0;
    s->simt.block_id[0] = warp->block_id[0];
    s->simt.block_id[1] = warp->block_id[1];
    s->simt.block_id[2] = warp->block_id[2];

    /* 首次调用时填充跳转表 */
    if (!dispatch_ready) {
        size_t di = 0;
        #define X(name, p, t, imm_fn) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
        #undef X
        dispatch_ready = 1;
    }

    /* 预译码 kernel */
    uint32_t kern_size = 4096;
    int tcount = 0;
    ThOp *code = threaded_predecode(s, s->kernel.kernel_addr, kern_size, &tcount);
    ThOp *ip = code;

    /* 解决 NULL handler */
    for (int _i = 0; _i <= tcount; _i++) {
        if (code[_i].handler == NULL)
            code[_i].handler = (_i < tcount) ? &&op_illegal : &&op_done;
    }

    goto *ip++->handler;

    /* ============ 指令实现 ============ */

op_jal: {
        
    l->gpr[ip[-1].rd].u32 = l->pc + 4;
    l->pc += ip[-1].imm;
    if (cycles++ > 100000000) return -1;
    ip = &code[ip[-1].branch_tgt];
    
    goto *ip++->handler;
}
op_jalr: {
        
    int32_t t = (l->gpr[ip[-1].rs1].u32 + ip[-1].imm) & ~1;
    l->gpr[ip[-1].rd].u32 = l->pc + 4;
    l->pc = t;
    if (cycles++ > 100000000) return -1;
    ip = &code[(t - s->kernel.kernel_addr) / 4];
    
    goto *ip++->handler;
}
op_beq: {
        
    if (l->gpr[ip[-1].rs1].u32 == l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_bne: {
        
    if (l->gpr[ip[-1].rs1].u32 != l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_blt: {
        
    if ((int32_t)l->gpr[ip[-1].rs1].u32 < (int32_t)l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_bge: {
        
    if ((int32_t)l->gpr[ip[-1].rs1].u32 >= (int32_t)l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_bltu: {
        
    if (l->gpr[ip[-1].rs1].u32 < l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_bgeu: {
        
    if (l->gpr[ip[-1].rs1].u32 >= l->gpr[ip[-1].rs2].u32)
        { ip = &code[ip[-1].branch_tgt]; }
    else { l->pc += 4; }
    if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

op_lb: {
        
    uint32_t a = l->gpr[ip[-1].rs1].u32 + ip[-1].imm;
    l->gpr[ip[-1].rd].i32 = (int32_t)(gpu_read(s, a, 1) << 24) >> 24;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_lh: {
        
    uint32_t a = l->gpr[ip[-1].rs1].u32 + ip[-1].imm;
    l->gpr[ip[-1].rd].i32 = (int32_t)(gpu_read(s, a, 2) << 16) >> 16;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_lw: {
        
    l->gpr[ip[-1].rd].u32 = gpu_read(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 4);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_lbu: {
        
    l->gpr[ip[-1].rd].u32 = gpu_read(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 1);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_lhu: {
        
    l->gpr[ip[-1].rd].u32 = gpu_read(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 2);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sb: {
        
    gpu_write(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 1, l->gpr[ip[-1].rs2].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sh: {
        
    gpu_write(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 2, l->gpr[ip[-1].rs2].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sw: {
        
    gpu_write(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 4, l->gpr[ip[-1].rs2].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_lui: {
        
    l->gpr[ip[-1].rd].u32 = ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_auipc: {
        
    l->gpr[ip[-1].rd].u32 = l->pc + ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_addi: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 + ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_slti: {
        
    l->gpr[ip[-1].rd].u32 = ((int32_t)l->gpr[ip[-1].rs1].u32 < ip[-1].imm) ? 1 : 0;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sltiu: {
        
    l->gpr[ip[-1].rd].u32 = (l->gpr[ip[-1].rs1].u32 < (uint32_t)ip[-1].imm) ? 1 : 0;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_xori: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 ^ ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_ori: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 | ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_andi: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 & ip[-1].imm;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_slli: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 << (ip[-1].imm & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_srli: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 >> (ip[-1].imm & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_srai: {
        
    l->gpr[ip[-1].rd].i32 = (int32_t)l->gpr[ip[-1].rs1].u32 >> (ip[-1].imm & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_add: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 + l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sub: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 - l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sll: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 << (l->gpr[ip[-1].rs2].u32 & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_slt: {
        
    l->gpr[ip[-1].rd].u32 = ((int32_t)l->gpr[ip[-1].rs1].u32 < (int32_t)l->gpr[ip[-1].rs2].u32) ? 1 : 0;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sltu: {
        
    l->gpr[ip[-1].rd].u32 = (l->gpr[ip[-1].rs1].u32 < l->gpr[ip[-1].rs2].u32) ? 1 : 0;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_xor: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 ^ l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_srl: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 >> (l->gpr[ip[-1].rs2].u32 & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_sra: {
        
    l->gpr[ip[-1].rd].i32 = (int32_t)l->gpr[ip[-1].rs1].u32 >> (l->gpr[ip[-1].rs2].u32 & 0x1F);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_or: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 | l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_and: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 & l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

/* RV32M */
op_mul: {
        
    l->gpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32 * l->gpr[ip[-1].rs2].u32;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_mulh: {
        
    l->gpr[ip[-1].rd].u32 = (uint32_t)(((int64_t)(int32_t)l->gpr[ip[-1].rs1].u32 * (int64_t)(int32_t)l->gpr[ip[-1].rs2].u32) >> 32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_mulhsu: {
        
    l->gpr[ip[-1].rd].u32 = (uint32_t)(((int64_t)(int32_t)l->gpr[ip[-1].rs1].u32 * (uint64_t)l->gpr[ip[-1].rs2].u32) >> 32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_mulhu: {
        
    l->gpr[ip[-1].rd].u32 = (uint32_t)(((uint64_t)l->gpr[ip[-1].rs1].u32 * (uint64_t)l->gpr[ip[-1].rs2].u32) >> 32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_div: {
        
    int32_t s1=l->gpr[ip[-1].rs1].i32, s2=l->gpr[ip[-1].rs2].i32;
    l->gpr[ip[-1].rd].i32 = (s2==0)?-1:s1/s2;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_divu: {
        
    uint32_t s1=l->gpr[ip[-1].rs1].u32, s2=l->gpr[ip[-1].rs2].u32;
    l->gpr[ip[-1].rd].u32 = (s2==0)?0xFFFFFFFF:s1/s2;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_rem: {
        
    int32_t s1=l->gpr[ip[-1].rs1].i32, s2=l->gpr[ip[-1].rs2].i32;
    l->gpr[ip[-1].rd].i32 = (s2==0)?s1:s1%s2;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_remu: {
        
    uint32_t s1=l->gpr[ip[-1].rs1].u32, s2=l->gpr[ip[-1].rs2].u32;
    l->gpr[ip[-1].rd].u32 = (s2==0)?s1:s1%s2;
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_ebreak: { return 0; }

/* CSR */
op_csrrw: {
    uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_read(l,c);
    l->gpr[ip[-1].rd].u32=o; csr_write(l,c,l->gpr[ip[-1].rs1].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    goto *ip++->handler;
}
op_csrrs: {
        
    uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_read(l,c);
    l->gpr[ip[-1].rd].u32=o; if(ip[-1].rs1!=0)csr_write(l,c,o|l->gpr[ip[-1].rs1].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_csrrc: {
        
    uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_read(l,c);
    l->gpr[ip[-1].rd].u32=o; if(ip[-1].rs1!=0)csr_write(l,c,o&~l->gpr[ip[-1].rs1].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_csrrwi: {
        
    uint16_t c=(uint16_t)ip[-1].imm;
    l->gpr[ip[-1].rd].u32=csr_read(l,c); csr_write(l,c,(uint32_t)ip[-1].rs1);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_csrrsi: {
        
    uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_read(l,c);
    l->gpr[ip[-1].rd].u32=o; if(ip[-1].rs1!=0)csr_write(l,c,o|(uint32_t)ip[-1].rs1);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}
op_csrrci: {
        
    uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_read(l,c);
    l->gpr[ip[-1].rd].u32=o; if(ip[-1].rs1!=0)csr_write(l,c,o&~(uint32_t)ip[-1].rs1);
    l->pc += 4; if (cycles++ > 100000000) return -1;
    
    goto *ip++->handler;
}

/* RV32F */
#define FP_HEAD() FP_SYNC()
#define FP_TAIL() do { FP_BACK(); l->pc += 4; if (cycles++ > 100000000) return -1; goto *ip++->handler; } while(0)

op_flw: {
        
    l->fpr[ip[-1].rd].u32 = gpu_read(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 4);
    l->pc += 4; if (cycles++ > 100000000) return -1; 
    goto *ip++->handler;
}
op_fsw: {
        
    gpu_write(s, l->gpr[ip[-1].rs1].u32 + ip[-1].imm, 4, l->fpr[ip[-1].rs2].u32);
    l->pc += 4; if (cycles++ > 100000000) return -1; 
    goto *ip++->handler;
}
op_fadd_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_add(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fsub_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_sub(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fmul_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_mul(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fdiv_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_div(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fsqrt_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_sqrt(l->fpr[ip[-1].rs1].u32, &l->fp_status);
    FP_TAIL();
}
op_fmadd_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_muladd(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, l->fpr[ip[-1].rs3].u32, 0, &l->fp_status);
    FP_TAIL();
}
op_fmsub_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_muladd(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, l->fpr[ip[-1].rs3].u32, float_muladd_negate_c, &l->fp_status);
    FP_TAIL();
}
op_fnmsub_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_muladd(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, l->fpr[ip[-1].rs3].u32, float_muladd_negate_product, &l->fp_status);
    FP_TAIL();
}
op_fnmadd_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_muladd(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, l->fpr[ip[-1].rs3].u32, float_muladd_negate_result, &l->fp_status);
    FP_TAIL();
}
op_fsgnj_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = (l->fpr[ip[-1].rs1].u32 & ~0x80000000) | (l->fpr[ip[-1].rs2].u32 & 0x80000000);
    FP_TAIL();
}
op_fsgnjn_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = (l->fpr[ip[-1].rs1].u32 & ~0x80000000) | ((~l->fpr[ip[-1].rs2].u32) & 0x80000000);
    FP_TAIL();
}
op_fsgnjx_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = l->fpr[ip[-1].rs1].u32 ^ (l->fpr[ip[-1].rs2].u32 & 0x80000000);
    FP_TAIL();
}
op_fmin_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_min(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fmax_s: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = float32_max(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_feq_s: { FP_HEAD();
    l->gpr[ip[-1].rd].u32 = float32_eq(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_flt_s: { FP_HEAD();
    l->gpr[ip[-1].rd].u32 = float32_lt(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fle_s: { FP_HEAD();
    l->gpr[ip[-1].rd].u32 = float32_le(l->fpr[ip[-1].rs1].u32, l->fpr[ip[-1].rs2].u32, &l->fp_status);
    FP_TAIL();
}
op_fcvt_w_s: { FP_HEAD(); {
    float32 _f = {l->fpr[ip[-1].rs1].u32};
    if (float32_is_quiet_nan(_f,&l->fp_status)||float32_is_signaling_nan(_f,&l->fp_status))
        l->gpr[ip[-1].rd].i32 = 0x7FFFFFFF;
    else {
        float32 _mx=int32_to_float32(0x7FFFFFFF,&l->fp_status),_mn=int32_to_float32(0x80000000,&l->fp_status);
        if(float32_le(_mx,_f,&l->fp_status))l->gpr[ip[-1].rd].i32=0x7FFFFFFF;
        else if(float32_lt(_f,_mn,&l->fp_status))l->gpr[ip[-1].rd].i32=0x80000000;
        else l->gpr[ip[-1].rd].i32=float32_to_int32(_f,&l->fp_status);
    }
    FP_TAIL();
}}
op_fcvt_wu_s: { FP_HEAD(); {
    float32 _f={l->fpr[ip[-1].rs1].u32};
    if(float32_is_quiet_nan(_f,&l->fp_status)||float32_is_signaling_nan(_f,&l->fp_status))
        l->gpr[ip[-1].rd].u32=0xFFFFFFFF;
    else if(float32_lt(_f,0,&l->fp_status))l->gpr[ip[-1].rd].u32=0;
    else{float32 _mu=uint32_to_float32(0xFFFFFFFF,&l->fp_status);
        if(float32_le(_mu,_f,&l->fp_status))l->gpr[ip[-1].rd].u32=0xFFFFFFFF;
        else l->gpr[ip[-1].rd].u32=float32_to_uint32(_f,&l->fp_status);}
    FP_TAIL();
}}
op_fcvt_s_w: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = int32_to_float32(l->gpr[ip[-1].rs1].i32, &l->fp_status);
    FP_TAIL();
}
op_fcvt_s_wu: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = uint32_to_float32(l->gpr[ip[-1].rs1].u32, &l->fp_status);
    FP_TAIL();
}
op_fmv_w_x: { FP_HEAD();
    l->fpr[ip[-1].rd].u32 = l->gpr[ip[-1].rs1].u32;
    FP_TAIL();
}
op_fmv_x_w: { FP_HEAD();
    l->gpr[ip[-1].rd].u32 = l->fpr[ip[-1].rs1].u32;
    FP_TAIL();
}
op_fclass_s: { FP_HEAD(); {
    uint32_t b=l->fpr[ip[-1].rs1].u32,e=(b>>23)&0xFF,m=b&0x7FFFFF,s=(b>>31)&1;int r=0;
    if(e==0xFF){if(m==0)r=s?(1<<0):(1<<7);else r=s?(1<<9):(1<<8);}
    else if(e==0){if(m==0)r=s?(1<<3):(1<<4);else r=s?(1<<2):(1<<5);}
    else{r=s?(1<<1):(1<<6);}l->gpr[ip[-1].rd].u32=r;
    FP_TAIL();
}}

/* LP float / scientific */
op_fcvt_s_bf16: { FP_HEAD(); l->fpr[ip[-1].rd].u32=bf16_to_f32(l->fpr[ip[-1].rs1].bf16); FP_TAIL(); }
op_fcvt_bf16_s: { FP_HEAD(); l->fpr[ip[-1].rd].bf16=f32_to_bf16(l->fpr[ip[-1].rs1].u32); FP_TAIL(); }
op_fcvt_s_e4m3: { FP_HEAD(); l->fpr[ip[-1].rd].u32=e4m3_to_f32(l->fpr[ip[-1].rs1].e4m3); FP_TAIL(); }
op_fcvt_e4m3_s: { FP_HEAD(); l->fpr[ip[-1].rd].e4m3=f32_to_e4m3(l->fpr[ip[-1].rs1].u32); FP_TAIL(); }
op_fcvt_s_e5m2: { FP_HEAD(); l->fpr[ip[-1].rd].u32=e5m2_to_f32(l->fpr[ip[-1].rs1].e5m2); FP_TAIL(); }
op_fcvt_e5m2_s: { FP_HEAD(); l->fpr[ip[-1].rd].e5m2=f32_to_e5m2(l->fpr[ip[-1].rs1].u32); FP_TAIL(); }
op_fcvt_s_e2m1: { FP_HEAD(); l->fpr[ip[-1].rd].u32=e2m1_to_f32(l->fpr[ip[-1].rs1].e2m1); FP_TAIL(); }
op_fcvt_e2m1_s: { FP_HEAD(); l->fpr[ip[-1].rd].e2m1=f32_to_e2m1(l->fpr[ip[-1].rs1].u32); FP_TAIL(); }

#define FP_SCALAR_T(name, expr) \
op_##name: { FP_HEAD(); { union{uint32_t u;float f;}_v; _v.u=l->fpr[ip[-1].rs1].u32; _v.f=expr; l->fpr[ip[-1].rd].u32=_v.u; } FP_TAIL(); }
FP_SCALAR_T(fexp_s, expf(_v.f))
FP_SCALAR_T(fln_s, logf(_v.f))
FP_SCALAR_T(frcp_s, 1.0f/_v.f)
FP_SCALAR_T(frsqrt_s, 1.0f/sqrtf(_v.f))
FP_SCALAR_T(ftanh_s, tanhf(_v.f))
FP_SCALAR_T(fsigmoid_s, 1.0f/(1.0f+expf(-_v.f)))
FP_SCALAR_T(fsin_s, sinf(_v.f))
FP_SCALAR_T(fcos_s, cosf(_v.f))
#undef FP_SCALAR_T

op_illegal: { return -1; }
op_done:   { return 0; }
}

/* ============================================================
 * Kernel dispatch（与原始相同）
 * ============================================================ */
int gpgpu_core_threaded_exec_kernel(GPGPUState *s)
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

                    gpgpu_core_init_warp(&warp, kernel_addr, tid_base, block_id, n_threads, w, blk_linear);

                    int ret = gpgpu_core_threaded_exec_warp(s, &warp, 100000000);
                    if (ret != 0) return -1;
                }
            }
        }
    }
    return 0;
}

/* warp init — 复刻原始版本 */
void gpgpu_core_threaded_init_warp(GPGPUWarp *warp, uint32_t pc,
                                    uint32_t tid_base, const uint32_t block_id[3],
                                    uint32_t num_threads, uint32_t warp_id, uint32_t blk_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = tid_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0]; warp->block_id[1] = block_id[1]; warp->block_id[2] = block_id[2];
    if (num_threads >= GPGPU_WARP_SIZE) warp->active_mask = 0xFFFFFFFF;
    else warp->active_mask = (1U << num_threads) - 1;
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(blk_linear, warp_id, i);
        lane->active = (warp->active_mask & (1 << i)) != 0;
        lane->gpr[0].u32 = 0; lane->fpr[0].u32 = 0;
        lane->fp_status.float_exception_flags |= float_flag_inexact;
        lane->fp_status.float_rounding_mode = float_round_nearest_even;
        lane->fp_status.default_nan_pattern = 0x7f;
    }
}
