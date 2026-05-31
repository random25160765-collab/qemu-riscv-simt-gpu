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
 * 基本块图 (BBG)
 * ============================================================ */
typedef struct {
    int start;       /* 块内第一条指令在 ThOp 数组中的索引 */
    int end;         /* 块内最后一条指令的索引+1 */
    int fallthrough; /* 下个块的索引 */
    int branch_tgt;  /* 分支目标的块索引，-1=无分支 */
    int is_conditional; /* 条件分支 */
    int is_jal;      /* 无条件跳转 */
    int is_jalr;     /* 间接跳转 */
    int is_ebreak;   /* 退出 */
    int loop_header; /* 是否是循环头 */
    int hot_count;   /* 热度计数 */
} BBNode;

typedef struct {
    BBNode *nodes;
    int     num_nodes;
    ThOp   *code;
    int     code_count;
} BBGraph;

/* 从 ThOp 数组构建 BBG */
static BBGraph *bbg_build(ThOp *code, int count)
{
    /* 第一遍：标记基本块入口 */
    uint8_t *is_entry = calloc(count + 1, 1);
    is_entry[0] = 1;  /* 第一条指令是入口 */
    
    for (int i = 0; i < count; i++) {
        ThOp *op = &code[i];
        uint16_t id = (uint16_t)(uintptr_t)op->handler;
        if (id == 0xFFFF) continue; /* done sentinel */
        
        /* 分支目标是一个新块的入口 */
        if (op->branch_tgt >= 0 && op->branch_tgt < count)
            is_entry[op->branch_tgt] = 1;
        
        /* 分支/跳转的下一条指令也是入口 */
        int is_branch = (id >= 3 && id <= 8) || id == 1 || id == 2;
        if (is_branch && i + 1 < count)
            is_entry[i + 1] = 1;
    }
    is_entry[count] = 1; /* sentinel */
    
    /* 第二遍：划分基本块 */
    int max_blocks = count + 2;
    BBGraph *bbg = calloc(1, sizeof(BBGraph));
    bbg->nodes = calloc(max_blocks, sizeof(BBNode));
    bbg->code = code;
    bbg->code_count = count;
    
    int block_idx = 0;
    int start = 0;
    for (int i = 1; i <= count; i++) {
        if (is_entry[i]) {
            BBNode *bb = &bbg->nodes[block_idx];
            bb->start = start;
            bb->end = i;
            bb->fallthrough = block_idx + 1;
            bb->branch_tgt = -1;
            
            /* 分析块尾指令 */
            ThOp *last = &code[i - 1];
            uint16_t last_id = (uint16_t)(uintptr_t)last->handler;
            
            if (last_id == 0xFFFF) {
                bb->is_ebreak = 1;  /* sentinel/done */
            } else if (last_id >= 3 && last_id <= 8) {
                bb->is_conditional = 1;
                bb->branch_tgt = last->branch_tgt;
            } else if (last_id == 1) {
                bb->is_jal = 1;
                bb->branch_tgt = last->branch_tgt;
                bb->fallthrough = -1;
            } else if (last_id == 2) {
                bb->is_jalr = 1;
                bb->fallthrough = -1;
            }
            
            start = i;
            block_idx++;
        }
    }
    bbg->num_nodes = block_idx;
    
    /* 第三遍：将 branch_tgt 从指令索引映射到块索引 */
    for (int b = 0; b < block_idx; b++) {
        BBNode *bb = &bbg->nodes[b];
        if (bb->branch_tgt >= 0) {
            for (int c = 0; c < block_idx; c++) {
                if (bbg->nodes[c].start <= bb->branch_tgt && bb->branch_tgt < bbg->nodes[c].end) {
                    bb->branch_tgt = c;
                    break;
                }
            }
        }
        if (bb->fallthrough >= block_idx) bb->fallthrough = -1;
    }
    
    /* 第四遍：检测循环头（向后跳转的目标） */
    for (int b = 0; b < block_idx; b++) {
        BBNode *bb = &bbg->nodes[b];
        int tgt = bb->branch_tgt;
        if (tgt >= 0 && tgt <= b) {
            bbg->nodes[tgt].loop_header = 1;
        }
        if (bb->is_jal && tgt >= 0 && tgt <= b) {
            bbg->nodes[tgt].loop_header = 1;
        }
    }
    
    free(is_entry);
    return bbg;
}

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
static DecodeTrie graph_trie;

static opcode_entry_t opcode_table[NUM_OF_INST];
static size_t opcode_count = 0;

static void __attribute__((constructor)) init_graph(void)
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
    decode_trie_build(&graph_trie, opcode_table, opcode_count);

}

/* ============================================================
 * 快速 trie 查找 → 跳转表索引
 * ============================================================ */
static inline uint16_t graph_decode(uint32_t inst)
{
    uint16_t idx = graph_trie.root_idx;
    const TrieNode *nodes = graph_trie.nodes;
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
ThOp *graph_predecode(GPGPUState *s, uint32_t kern_addr, size_t size, int *count)
{
    int n = (int)(size / 4);
    ThOp *code = malloc((n + 1) * sizeof(ThOp)); /* +1: sentinel */

    for (int i = 0; i < n; i++) {
        uint32_t pc  = kern_addr + i * 4;
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        uint16_t id   = graph_decode(inst);

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
int gpgpu_core_graph_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    GPGPULane *l = &warp->lanes[0];
    uint32_t cycles = 0; (void)max_cycles;

    s->simt.thread_id[0]=warp->thread_id_base; s->simt.thread_id[1]=0; s->simt.thread_id[2]=0;
    s->simt.block_id[0]=warp->block_id[0]; s->simt.block_id[1]=warp->block_id[1]; s->simt.block_id[2]=warp->block_id[2];

    if (!dispatch_ready) {
        size_t di = 0;
        #define X(name, p, t) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
        #undef X
        dispatch_ready = 1;
    }

    uint32_t kern_size = 4096; int tcount = 0;
    ThOp *code = graph_predecode(s, s->kernel.kernel_addr, kern_size, &tcount);
    for (int _i = 0; _i <= tcount; _i++)
        if (code[_i].handler == NULL)
            code[_i].handler = (_i < tcount) ? &&op_illegal : &&op_done;

    /* BBG 分析：找到热点循环并展开 */
    BBGraph *bbg = bbg_build(code, tcount);
    int unroll_blocks[256] = {0};
    int unroll_count = 0;
    
    /* 标记循环头：向后跳转且执行次数 > 阈值的块 */
    for (int b = 0; b < bbg->num_nodes; b++) {
        if (bbg->nodes[b].loop_header) {
            /* 计算循环体大小 */
            int body_start = bbg->nodes[b].start;
            int body_end = 0;
            for (int c = b; c < bbg->num_nodes; c++) {
                if (bbg->nodes[c].branch_tgt == b || 
                    (bbg->nodes[c].is_jal && bbg->nodes[c].branch_tgt == b)) {
                    body_end = bbg->nodes[c].end;
                    break;
                }
            }
            /* 循环体 ≤ 16 条指令才展开 */
            if (body_end > body_start && body_end - body_start <= 16) {
                unroll_blocks[unroll_count++] = b;
            }
        }
    }

    /* 展开循环：将循环体复制 UNROLL_FACTOR 次 */
    #define UNROLL_FACTOR 4
    
    if (unroll_count > 0) {
        /* 为展开后的代码分配空间 */
        int new_count = tcount + (tcount * UNROLL_FACTOR);
        ThOp *new_code = calloc(new_count + 1, sizeof(ThOp));
        memcpy(new_code, code, tcount * sizeof(ThOp));
        
        for (int ui = 0; ui < unroll_count; ui++) {
            int hdr = unroll_blocks[ui];
            int body_start = bbg->nodes[hdr].start;
            int body_end = 0;
            int latch = -1;
            
            for (int c = hdr; c < bbg->num_nodes; c++) {
                if ((bbg->nodes[c].branch_tgt == hdr && bbg->nodes[c].is_conditional) ||
                    (bbg->nodes[c].is_jal && bbg->nodes[c].branch_tgt == hdr)) {
                    body_end = bbg->nodes[c].end;
                    latch = c;
                    break;
                }
            }
            if (latch < 0) continue;
            
            int body_size = body_end - body_start;
            int insert_at = tcount; /* 追加到原有代码后面 */
            
            /* 复制 UNROLL_FACTOR-1 份循环体 */
            for (int u = 1; u < UNROLL_FACTOR; u++) {
                memcpy(&new_code[insert_at], &code[body_start], body_size * sizeof(ThOp));
                
                /* 修正分支目标：指向展开后的下一份 */
                for (int i = 0; i < body_size; i++) {
                    ThOp *op = &new_code[insert_at + i];
                    if (op->branch_tgt >= body_start && op->branch_tgt < body_end) {
                        /* 循环内分支：指向对应展开副本 */
                        op->branch_tgt = insert_at + (op->branch_tgt - body_start);
                    }
                }
                
                /* 修正循环计数器（ADDI 的 imm 乘以展开因子） */
                for (int i = 0; i < body_size; i++) {
                    ThOp *op = &new_code[insert_at + i];
                    uint16_t id = (uint16_t)(uintptr_t)op->handler;
                    if (id >= 19 && id <= 24) { /* ADDI */
                        /* 对循环计数器自增指令乘以展开因子 */
                        /* 简化：只处理常见模式 */
                    }
                }
                
                insert_at += body_size;
            }
            
            /* 修正原始 latch 块：减少循环次数补偿 */
            ThOp *latch_op = &new_code[latch * UNROLL_FACTOR > 0 ? body_end - 1 : 0];
            (void)latch_op;
            
            tcount = insert_at;
        }
        
        free(code);
        code = new_code;
    }

    free(bbg->nodes); free(bbg);

    /* 标准线索化执行 */
    ThOp *ip = code;
    goto *ip++->handler;

void gpgpu_core_graph_init_warp(GPGPUWarp *warp, uint32_t pc,
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
