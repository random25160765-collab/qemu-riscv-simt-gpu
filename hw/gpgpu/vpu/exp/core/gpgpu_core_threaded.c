/*
 * gpgpu_core_threaded.c — 线索化解释器
 *   编译时预译码 kernel → ThOp 数组
 *   运行时: goto *ip++->handler 零开销循环
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "state.h"
#include "gpgpu_core.h"
#include "memory.h"
#include "inst.h"
#include "utils.h"
#include "proto.h"
#include "decode_trie.h"
#include "lpfp.h"

typedef enum { TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B, TYPE_CSR, TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4 } inst_type_t;
typedef struct {
    void    *handler;
    uint32_t inst;
    int32_t  imm;
    int8_t   rd, rs1, rs2, rs3;
    int32_t  branch_tgt;
} ThOp;

#define NUM_OF_INST 300
static void *dispatch[NUM_OF_INST];
static int dispatch_ready = 0;
static DecodeTrie trie;
static opcode_entry_t op_table[NUM_OF_INST];
static size_t op_count = 0;


#define INSTRUCTION_LIST \
    X(jal,      "??????? ????? ????? ??? ????? 11011 11", TYPE_J); \
    X(jalr,     "??????? ????? ????? 000 ????? 11001 11", TYPE_I); \
    X(beq,      "??????? ????? ????? 000 ????? 11000 11", TYPE_B); \
    X(bne,      "??????? ????? ????? 001 ????? 11000 11", TYPE_B); \
    X(blt,      "??????? ????? ????? 100 ????? 11000 11", TYPE_B); \
    X(bge,      "??????? ????? ????? 101 ????? 11000 11", TYPE_B); \
    X(bltu,     "??????? ????? ????? 110 ????? 11000 11", TYPE_B); \
    X(bgeu,     "??????? ????? ????? 111 ????? 11000 11", TYPE_B); \
    X(lb,       "??????? ????? ????? 000 ????? 00000 11", TYPE_I); \
    X(lh,       "??????? ????? ????? 001 ????? 00000 11", TYPE_I); \
    X(lw,       "??????? ????? ????? 010 ????? 00000 11", TYPE_I); \
    X(lbu,      "??????? ????? ????? 100 ????? 00000 11", TYPE_I); \
    X(lhu,      "??????? ????? ????? 101 ????? 00000 11", TYPE_I); \
    X(sb,       "??????? ????? ????? 000 ????? 01000 11", TYPE_S); \
    X(sh,       "??????? ????? ????? 001 ????? 01000 11", TYPE_S); \
    X(sw,       "??????? ????? ????? 010 ????? 01000 11", TYPE_S); \
    X(lui,      "??????? ????? ????? ??? ????? 01101 11", TYPE_U); \
    X(auipc,    "??????? ????? ????? ??? ????? 00101 11", TYPE_U); \
    X(addi,     "??????? ????? ????? 000 ????? 00100 11", TYPE_I); \
    X(slti,     "??????? ????? ????? 010 ????? 00100 11", TYPE_I); \
    X(sltiu,    "??????? ????? ????? 011 ????? 00100 11", TYPE_I); \
    X(xori,     "??????? ????? ????? 100 ????? 00100 11", TYPE_I); \
    X(ori,      "??????? ????? ????? 110 ????? 00100 11", TYPE_I); \
    X(andi,     "??????? ????? ????? 111 ????? 00100 11", TYPE_I); \
    X(slli,     "0000000 ????? ????? 001 ????? 00100 11", TYPE_I); \
    X(srli,     "0000000 ????? ????? 101 ????? 00100 11", TYPE_I); \
    X(srai,     "0100000 ????? ????? 101 ????? 00100 11", TYPE_I); \
    X(add,      "0000000 ????? ????? 000 ????? 01100 11", TYPE_R); \
    X(sub,      "0100000 ????? ????? 000 ????? 01100 11", TYPE_R); \
    X(sll,      "0000000 ????? ????? 001 ????? 01100 11", TYPE_R); \
    X(slt,      "0000000 ????? ????? 010 ????? 01100 11", TYPE_R); \
    X(sltu,     "0000000 ????? ????? 011 ????? 01100 11", TYPE_R); \
    X(xor,      "0000000 ????? ????? 100 ????? 01100 11", TYPE_R); \
    X(srl,      "0000000 ????? ????? 101 ????? 01100 11", TYPE_R); \
    X(sra,      "0100000 ????? ????? 101 ????? 01100 11", TYPE_R); \
    X(or,       "0000000 ????? ????? 110 ????? 01100 11", TYPE_R); \
    X(and,      "0000000 ????? ????? 111 ????? 01100 11", TYPE_R); \
    X(mul,      "0000001 ????? ????? 000 ????? 01100 11", TYPE_R); \
    X(mulh,     "0000001 ????? ????? 001 ????? 01100 11", TYPE_R); \
    X(mulhsu,   "0000001 ????? ????? 010 ????? 01100 11", TYPE_R); \
    X(mulhu,    "0000001 ????? ????? 011 ????? 01100 11", TYPE_R); \
    X(div,      "0000001 ????? ????? 100 ????? 01100 11", TYPE_R); \
    X(divu,     "0000001 ????? ????? 101 ????? 01100 11", TYPE_R); \
    X(rem,      "0000001 ????? ????? 110 ????? 01100 11", TYPE_R); \
    X(remu,     "0000001 ????? ????? 111 ????? 01100 11", TYPE_R); \
    X(csrrw,    "??????? ????? ????? 001 ????? 11100 11", TYPE_CSR); \
    X(csrrs,    "??????? ????? ????? 010 ????? 11100 11", TYPE_CSR); \
    X(csrrc,    "??????? ????? ????? 011 ????? 11100 11", TYPE_CSR); \
    X(csrrwi,   "??????? ????? ????? 101 ????? 11100 11", TYPE_CSR); \
    X(csrrsi,   "??????? ????? ????? 110 ????? 11100 11", TYPE_CSR); \
    X(csrrci,   "??????? ????? ????? 111 ????? 11100 11", TYPE_CSR); \
    X(ebreak,   "0000000 00001 00000 000 00000 11100 11", TYPE_I); \
    X(flw,      "??????? ????? ????? 010 ????? 00001 11", TYPE_I); \
    X(fsw,      "??????? ????? ????? 010 ????? 01001 11", TYPE_S); \
    X(fmadd_s,  "?????00 ????? ????? ??? ????? 10000 11", TYPE_F4); \
    X(fmsub_s,  "?????00 ????? ????? ??? ????? 10001 11", TYPE_F4); \
    X(fnmsub_s, "?????00 ????? ????? ??? ????? 10010 11", TYPE_F4); \
    X(fnmadd_s, "?????00 ????? ????? ??? ????? 10011 11", TYPE_F4); \
    X(fadd_s,   "0000000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsub_s,   "0000100 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fmul_s,   "0001000 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fdiv_s,   "0001100 ????? ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsqrt_s,  "0101100 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsgnj_s,  "0010000 ????? ????? 000 ????? 10100 11", TYPE_FR); \
    X(fsgnjn_s, "0010000 ????? ????? 001 ????? 10100 11", TYPE_FR); \
    X(fsgnjx_s, "0010000 ????? ????? 010 ????? 10100 11", TYPE_FR); \
    X(fmin_s,   "0010100 ????? ????? 000 ????? 10100 11", TYPE_FR); \
    X(fmax_s,   "0010100 ????? ????? 001 ????? 10100 11", TYPE_FR); \
    X(fcvt_w_s, "1100000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_wu_s,"1100000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fmv_x_w,  "1110000 00000 ????? 000 ????? 10100 11", TYPE_FR); \
    X(feq_s,    "1010000 ????? ????? 010 ????? 10100 11", TYPE_FR); \
    X(flt_s,    "1010000 ????? ????? 001 ????? 10100 11", TYPE_FR); \
    X(fle_s,    "1010000 ????? ????? 000 ????? 10100 11", TYPE_FR); \
    X(fclass_s, "1110000 00000 ????? 001 ????? 10100 11", TYPE_FR); \
    X(fcvt_s_w, "1101000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_wu,"1101000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fmv_w_x,  "1111000 00000 ????? 000 ????? 10100 11", TYPE_FR); \
    X(fcvt_s_bf16,"0100010 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_bf16_s,"0100010 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e4m3,"0100100 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e4m3_s,"0100100 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e5m2,"0100100 00010 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e5m2_s,"0100100 00011 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_s_e2m1,"0100110 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcvt_e2m1_s,"0100110 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fexp_s,   "0110000 00000 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fln_s,    "0110000 00001 ????? ??? ????? 10100 11", TYPE_FR); \
    X(frcp_s,   "0110000 00010 ????? ??? ????? 10100 11", TYPE_FR); \
    X(frsqrt_s, "0110000 00011 ????? ??? ????? 10100 11", TYPE_FR); \
    X(ftanh_s,  "0110000 00100 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsigmoid_s,"0110000 00101 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fsin_s,   "0110000 00110 ????? ??? ????? 10100 11", TYPE_FR); \
    X(fcos_s,   "0110000 00111 ????? ??? ????? 10100 11", TYPE_FR);

static void __attribute__((constructor)) init_threaded(void)
{
    int idx = 0;
    #define X(name, pattern, op_type) \
        do { op_table[idx].mask=pattern_to_mask(pattern); op_table[idx].match=pattern_to_match(pattern); \
             op_table[idx].exec=NULL; op_table[idx].type=op_type; idx++; } while(0)
    INSTRUCTION_LIST
    #undef X
    op_count = idx;
    decode_trie_build(&trie, op_table, op_count);
}

static inline uint16_t th_decode(uint32_t inst)
{
    uint16_t idx = trie.root_idx;
    const TrieNode *nodes = trie.nodes;
    while (nodes[idx].bit_pos >= 0) {
        int bit = (inst >> nodes[idx].bit_pos) & 1;
        uint16_t next = nodes[idx].child[bit];
        if (next == 0) return 0;
        idx = next;
    }
    return nodes[idx].instr_id;
}

ThOp *th_predecode(GPGPUState *s, uint32_t kern_addr, size_t size, int *count)
{
    int n = (int)(size / 4);
    ThOp *code = malloc((n + 1) * sizeof(ThOp));
    for (int i = 0; i < n; i++) {
        uint32_t pc = kern_addr + i * 4;
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        uint16_t id = th_decode(inst);
        code[i].inst = inst;
        code[i].rd  = (int8_t)BITS(inst, 11, 7);
        code[i].rs1 = (int8_t)BITS(inst, 19, 15);
        code[i].rs2 = (int8_t)BITS(inst, 24, 20);
        code[i].rs3 = (int8_t)BITS(inst, 31, 27);
        code[i].branch_tgt = -1;
        if (id == 0) { code[i].handler = NULL; continue; }
        code[i].handler = dispatch[id - 1];
        switch (inst & 0x7F) {
        case 0x37: case 0x17: code[i].imm = immU(inst); break;
        case 0x6F: code[i].imm = immJ(inst);
                    code[i].branch_tgt = (int)((pc + code[i].imm - kern_addr) / 4); break;
        case 0x67: code[i].imm = immI(inst); break;
        case 0x63: code[i].imm = immB(inst);
                    code[i].branch_tgt = (int)((pc + code[i].imm - kern_addr) / 4); break;
        case 0x03: case 0x13: code[i].imm = immI(inst); break;
        case 0x23: code[i].imm = immS(inst); break;
        case 0x33: code[i].imm = 0; break;
        case 0x73:
            if ((inst & 0xFE00707F) == 0x00100073) code[i].imm = 0;
            else code[i].imm = immCSR(inst); break;
        case 0x07: code[i].imm = immI(inst); break;
        case 0x27: code[i].imm = immS(inst); break;
        default:   code[i].imm = 0; break;
        }
    }
    code[n].handler = NULL; code[n].inst = 0;
    *count = n; return code;
}

static uint32_t csr_rd(GPGPULane *l, uint16_t a) {
    switch (a) { case CSR_MHARTID: return l->mhartid; case CSR_FFLAGS: return l->fcsr&0x1F;
    case CSR_FRM: return (l->fcsr>>5)&0x7; case CSR_FCSR: return l->fcsr; default: return 0; }
}
static void csr_wr(GPGPULane *l, uint16_t a, uint32_t v) {
    switch (a) { case CSR_FFLAGS: l->fcsr=(l->fcsr&~0x1F)|(v&0x1F); break;
    case CSR_FRM: l->fcsr=(l->fcsr&~0xE0)|((v&0x7)<<5); break; case CSR_FCSR: l->fcsr=v; break; default: break; }
}
#define FP_SYNC() do { uint8_t f_=(l->fcsr>>5)&0x7; l->fp_status.float_rounding_mode=(f_<=4)?f_:0; l->fp_status.float_exception_flags=float_flag_inexact; } while(0)
#define FP_BACK() do { l->fcsr=(l->fcsr&~0x1F)|(l->fp_status.float_exception_flags&0x1F); } while(0)

int gpgpu_core_threaded_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
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
    ThOp *code = th_predecode(s, s->kernel.kernel_addr, kern_size, &tcount);
    ThOp *ip = code;
    for (int _i = 0; _i <= tcount; _i++)
        if (code[_i].handler == NULL)
            code[_i].handler = (_i < tcount) ? &&op_illegal : &&op_done;

    goto *ip++->handler;

op_jal: {
    l->gpr[ip[-1].rd].u32 = l->pc + 4; l->pc += ip[-1].imm;
    if (++cycles > 100000000) return -1;
    ip = &code[ip[-1].branch_tgt]; goto *ip++->handler;
}
op_jalr: {
    int32_t t = (l->gpr[ip[-1].rs1].u32 + ip[-1].imm) & ~1;
    l->gpr[ip[-1].rd].u32 = l->pc + 4; l->pc = t;
    if (++cycles > 100000000) return -1;
    ip = &code[(t - s->kernel.kernel_addr) / 4]; goto *ip++->handler;
}
#define BR(name, cond) op_##name: { if (cond) { ip = &code[ip[-1].branch_tgt]; } else { l->pc += 4; } \
    if (++cycles > 100000000) return -1; goto *ip++->handler; }
BR(beq,  l->gpr[ip[-1].rs1].u32 == l->gpr[ip[-1].rs2].u32)
BR(bne,  l->gpr[ip[-1].rs1].u32 != l->gpr[ip[-1].rs2].u32)
BR(blt,  (int32_t)l->gpr[ip[-1].rs1].u32 <  (int32_t)l->gpr[ip[-1].rs2].u32)
BR(bge,  (int32_t)l->gpr[ip[-1].rs1].u32 >= (int32_t)l->gpr[ip[-1].rs2].u32)
BR(bltu, l->gpr[ip[-1].rs1].u32 <  l->gpr[ip[-1].rs2].u32)
BR(bgeu, l->gpr[ip[-1].rs1].u32 >= l->gpr[ip[-1].rs2].u32)
#undef BR

#define LD(name, field, expr) op_##name: { l->gpr[ip[-1].rd].field = expr; l->pc += 4; \
    if (++cycles > 100000000) return -1; goto *ip++->handler; }
LD(lb,  i32, (int32_t)(gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,1)<<24)>>24)
LD(lh,  i32, (int32_t)(gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,2)<<16)>>16)
LD(lw,  u32, gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,4))
LD(lbu, u32, gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,1))
LD(lhu, u32, gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,2))
#undef LD

op_flw: { l->fpr[ip[-1].rd].u32=gpu_read(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,4); l->pc+=4;
    if(++cycles>100000000)return -1; goto *ip++->handler; }

#define ST(name, sz, src) op_##name: { gpu_write(s,l->gpr[ip[-1].rs1].u32+ip[-1].imm,sz,src); l->pc+=4; \
    if(++cycles>100000000)return -1; goto *ip++->handler; }
ST(sb,1,l->gpr[ip[-1].rs2].u32)  ST(sh,2,l->gpr[ip[-1].rs2].u32)  ST(sw,4,l->gpr[ip[-1].rs2].u32)
ST(fsw,4,l->fpr[ip[-1].rs2].u32)
#undef ST

op_lui:  { l->gpr[ip[-1].rd].u32=ip[-1].imm; l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler; }
op_auipc:{ l->gpr[ip[-1].rd].u32=l->pc+ip[-1].imm; l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler; }

#define AI(name, expr) op_##name: { l->gpr[ip[-1].rd].u32=expr; l->pc+=4; \
    if(++cycles>100000000)return -1; goto *ip++->handler; }
AI(addi, l->gpr[ip[-1].rs1].u32+ip[-1].imm)
AI(slti, ((int32_t)l->gpr[ip[-1].rs1].u32<ip[-1].imm)?1:0)
AI(sltiu,(l->gpr[ip[-1].rs1].u32<(uint32_t)ip[-1].imm)?1:0)
AI(xori, l->gpr[ip[-1].rs1].u32^ip[-1].imm)
AI(ori,  l->gpr[ip[-1].rs1].u32|ip[-1].imm)
AI(andi, l->gpr[ip[-1].rs1].u32&ip[-1].imm)
AI(slli, l->gpr[ip[-1].rs1].u32<<(ip[-1].imm&0x1F))
AI(srli, l->gpr[ip[-1].rs1].u32>>(ip[-1].imm&0x1F))
AI(srai, (uint32_t)((int32_t)l->gpr[ip[-1].rs1].u32>>(ip[-1].imm&0x1F)))
#undef AI

#define AL(name, expr) op_##name: { l->gpr[ip[-1].rd].u32=expr; l->pc+=4; \
    if(++cycles>100000000)return -1; goto *ip++->handler; }
AL(add,  l->gpr[ip[-1].rs1].u32+l->gpr[ip[-1].rs2].u32)
AL(sub,  l->gpr[ip[-1].rs1].u32-l->gpr[ip[-1].rs2].u32)
AL(sll,  l->gpr[ip[-1].rs1].u32<<(l->gpr[ip[-1].rs2].u32&0x1F))
AL(slt,  ((int32_t)l->gpr[ip[-1].rs1].u32<(int32_t)l->gpr[ip[-1].rs2].u32)?1:0)
AL(sltu, (l->gpr[ip[-1].rs1].u32<l->gpr[ip[-1].rs2].u32)?1:0)
AL(xor,  l->gpr[ip[-1].rs1].u32^l->gpr[ip[-1].rs2].u32)
AL(srl,  l->gpr[ip[-1].rs1].u32>>(l->gpr[ip[-1].rs2].u32&0x1F))
AL(sra,  (uint32_t)((int32_t)l->gpr[ip[-1].rs1].u32>>(l->gpr[ip[-1].rs2].u32&0x1F)))
AL(or,   l->gpr[ip[-1].rs1].u32|l->gpr[ip[-1].rs2].u32)
AL(and,  l->gpr[ip[-1].rs1].u32&l->gpr[ip[-1].rs2].u32)
AL(mul,  l->gpr[ip[-1].rs1].u32*l->gpr[ip[-1].rs2].u32)
AL(mulh, (uint32_t)(((int64_t)(int32_t)l->gpr[ip[-1].rs1].u32*(int64_t)(int32_t)l->gpr[ip[-1].rs2].u32)>>32))
AL(mulhsu,(uint32_t)(((int64_t)(int32_t)l->gpr[ip[-1].rs1].u32*(uint64_t)l->gpr[ip[-1].rs2].u32)>>32))
AL(mulhu,(uint32_t)(((uint64_t)l->gpr[ip[-1].rs1].u32*(uint64_t)l->gpr[ip[-1].rs2].u32)>>32))
AL(div,  (uint32_t)(l->gpr[ip[-1].rs2].i32?l->gpr[ip[-1].rs1].i32/l->gpr[ip[-1].rs2].i32:-1))
AL(divu, l->gpr[ip[-1].rs2].u32?l->gpr[ip[-1].rs1].u32/l->gpr[ip[-1].rs2].u32:0xFFFFFFFF)
AL(rem,  (uint32_t)(l->gpr[ip[-1].rs2].i32?l->gpr[ip[-1].rs1].i32%l->gpr[ip[-1].rs2].i32:l->gpr[ip[-1].rs1].i32))
AL(remu, l->gpr[ip[-1].rs2].u32?l->gpr[ip[-1].rs1].u32%l->gpr[ip[-1].rs2].u32:l->gpr[ip[-1].rs1].u32)
#undef AL

op_ebreak: { return 0; }

#define CSR(name, body) op_##name: { uint16_t c=(uint16_t)ip[-1].imm; uint32_t o=csr_rd(l,c); \
    l->gpr[ip[-1].rd].u32=o; body; l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler; }
CSR(csrrw, csr_wr(l,c,l->gpr[ip[-1].rs1].u32))
CSR(csrrs, if(ip[-1].rs1!=0)csr_wr(l,c,o|l->gpr[ip[-1].rs1].u32))
CSR(csrrc, if(ip[-1].rs1!=0)csr_wr(l,c,o&~l->gpr[ip[-1].rs1].u32))
CSR(csrrwi,csr_wr(l,c,(uint32_t)ip[-1].rs1))
CSR(csrrsi,if(ip[-1].rs1!=0)csr_wr(l,c,o|(uint32_t)ip[-1].rs1))
CSR(csrrci,if(ip[-1].rs1!=0)csr_wr(l,c,o&~(uint32_t)ip[-1].rs1))
#undef CSR

#define FP(name, expr) op_##name: { FP_SYNC(); expr; FP_BACK(); l->pc+=4; \
    if(++cycles>100000000)return -1; goto *ip++->handler; }
FP(fadd_s, l->fpr[ip[-1].rd].u32=float32_add(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fsub_s, l->fpr[ip[-1].rd].u32=float32_sub(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fmul_s, l->fpr[ip[-1].rd].u32=float32_mul(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fdiv_s, l->fpr[ip[-1].rd].u32=float32_div(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fsqrt_s,l->fpr[ip[-1].rd].u32=float32_sqrt(l->fpr[ip[-1].rs1].u32,&l->fp_status))
FP(fmadd_s, l->fpr[ip[-1].rd].u32=float32_muladd(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,l->fpr[ip[-1].rs3].u32,0,&l->fp_status))
FP(fmsub_s, l->fpr[ip[-1].rd].u32=float32_muladd(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,l->fpr[ip[-1].rs3].u32,float_muladd_negate_c,&l->fp_status))
FP(fnmsub_s,l->fpr[ip[-1].rd].u32=float32_muladd(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,l->fpr[ip[-1].rs3].u32,float_muladd_negate_product,&l->fp_status))
FP(fnmadd_s,l->fpr[ip[-1].rd].u32=float32_muladd(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,l->fpr[ip[-1].rs3].u32,float_muladd_negate_result,&l->fp_status))
FP(fsgnj_s, l->fpr[ip[-1].rd].u32=(l->fpr[ip[-1].rs1].u32&~0x80000000)|(l->fpr[ip[-1].rs2].u32&0x80000000))
FP(fsgnjn_s,l->fpr[ip[-1].rd].u32=(l->fpr[ip[-1].rs1].u32&~0x80000000)|((~l->fpr[ip[-1].rs2].u32)&0x80000000))
FP(fsgnjx_s,l->fpr[ip[-1].rd].u32=l->fpr[ip[-1].rs1].u32^(l->fpr[ip[-1].rs2].u32&0x80000000))
FP(fmin_s, l->fpr[ip[-1].rd].u32=float32_min(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fmax_s, l->fpr[ip[-1].rd].u32=float32_max(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(feq_s, l->gpr[ip[-1].rd].u32=float32_eq(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(flt_s, l->gpr[ip[-1].rd].u32=float32_lt(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fle_s, l->gpr[ip[-1].rd].u32=float32_le(l->fpr[ip[-1].rs1].u32,l->fpr[ip[-1].rs2].u32,&l->fp_status))
FP(fcvt_s_w, l->fpr[ip[-1].rd].u32=int32_to_float32(l->gpr[ip[-1].rs1].i32,&l->fp_status))
FP(fcvt_s_wu,l->fpr[ip[-1].rd].u32=uint32_to_float32(l->gpr[ip[-1].rs1].u32,&l->fp_status))
FP(fmv_w_x, l->fpr[ip[-1].rd].u32=l->gpr[ip[-1].rs1].u32)
FP(fmv_x_w, l->gpr[ip[-1].rd].u32=l->fpr[ip[-1].rs1].u32)

op_fcvt_w_s: { FP_SYNC(); {
    float32 _f=l->fpr[ip[-1].rs1].u32;
    if(float32_is_quiet_nan(_f,&l->fp_status)||float32_is_signaling_nan(_f,&l->fp_status))
        l->gpr[ip[-1].rd].i32=0x7FFFFFFF;
    else{float32 _mx=int32_to_float32(0x7FFFFFFF,&l->fp_status),_mn=int32_to_float32(0x80000000,&l->fp_status);
    if(float32_le(_mx,_f,&l->fp_status))l->gpr[ip[-1].rd].i32=0x7FFFFFFF;
    else if(float32_lt(_f,_mn,&l->fp_status))l->gpr[ip[-1].rd].i32=0x80000000;
    else l->gpr[ip[-1].rd].i32=float32_to_int32(_f,&l->fp_status);}
    FP_BACK(); l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler;
}}
op_fcvt_wu_s: { FP_SYNC(); {
    float32 _f=l->fpr[ip[-1].rs1].u32;
    if(float32_is_quiet_nan(_f,&l->fp_status)||float32_is_signaling_nan(_f,&l->fp_status))
        l->gpr[ip[-1].rd].u32=0xFFFFFFFF;
    else if(float32_lt(_f,0,&l->fp_status))l->gpr[ip[-1].rd].u32=0;
    else{float32 _mu=uint32_to_float32(0xFFFFFFFF,&l->fp_status);
    if(float32_le(_mu,_f,&l->fp_status))l->gpr[ip[-1].rd].u32=0xFFFFFFFF;
    else l->gpr[ip[-1].rd].u32=float32_to_uint32(_f,&l->fp_status);}
    FP_BACK(); l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler;
}}
op_fclass_s: { FP_SYNC(); {
    uint32_t b=l->fpr[ip[-1].rs1].u32,e=(b>>23)&0xFF,m=b&0x7FFFFF,s=(b>>31)&1;int r=0;
    if(e==0xFF)r=m?(s?(1<<9):(1<<8)):(s?(1<<0):(1<<7));
    else if(e==0)r=m?(s?(1<<2):(1<<5)):(s?(1<<3):(1<<4));
    else r=s?(1<<1):(1<<6);
    l->gpr[ip[-1].rd].u32=r; FP_BACK(); l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler;
}}

FP(fcvt_s_bf16, l->fpr[ip[-1].rd].u32=bf16_to_f32(l->fpr[ip[-1].rs1].bf16))
FP(fcvt_bf16_s, l->fpr[ip[-1].rd].bf16=f32_to_bf16(l->fpr[ip[-1].rs1].u32))
FP(fcvt_s_e4m3, l->fpr[ip[-1].rd].u32=e4m3_to_f32(l->fpr[ip[-1].rs1].e4m3))
FP(fcvt_e4m3_s, l->fpr[ip[-1].rd].e4m3=f32_to_e4m3(l->fpr[ip[-1].rs1].u32))
FP(fcvt_s_e5m2, l->fpr[ip[-1].rd].u32=e5m2_to_f32(l->fpr[ip[-1].rs1].e5m2))
FP(fcvt_e5m2_s, l->fpr[ip[-1].rd].e5m2=f32_to_e5m2(l->fpr[ip[-1].rs1].u32))
FP(fcvt_s_e2m1, l->fpr[ip[-1].rd].u32=e2m1_to_f32(l->fpr[ip[-1].rs1].e2m1))
FP(fcvt_e2m1_s, l->fpr[ip[-1].rd].e2m1=f32_to_e2m1(l->fpr[ip[-1].rs1].u32))

#define SCI(name, expr) op_##name: { FP_SYNC(); { union{uint32_t u;float f;}_v; \
    _v.u=l->fpr[ip[-1].rs1].u32; _v.f=expr; l->fpr[ip[-1].rd].u32=_v.u; } \
    FP_BACK(); l->pc+=4; if(++cycles>100000000)return -1; goto *ip++->handler; }
SCI(fexp_s, expf(_v.f))
SCI(fln_s, logf(_v.f))
SCI(frcp_s, 1.0f/_v.f)
SCI(frsqrt_s, 1.0f/sqrtf(_v.f))
SCI(ftanh_s, tanhf(_v.f))
SCI(fsigmoid_s, 1.0f/(1.0f+expf(-_v.f)))
SCI(fsin_s, sinf(_v.f))
SCI(fcos_s, cosf(_v.f))
#undef SCI
#undef FP

op_illegal: { return -1; }
op_done:    { return 0; }
}

void gpgpu_core_threaded_init_warp(GPGPUWarp *warp, uint32_t pc,
    uint32_t tid_base, const uint32_t block_id[3], uint32_t num_threads,
    uint32_t warp_id, uint32_t blk_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = tid_base; warp->warp_id = warp_id;
    warp->block_id[0]=block_id[0]; warp->block_id[1]=block_id[1]; warp->block_id[2]=block_id[2];
    warp->active_mask = (num_threads >= GPGPU_WARP_SIZE) ? 0xFFFFFFFF : (1U << num_threads) - 1;
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc; lane->mhartid = MHARTID_ENCODE(blk_linear, warp_id, i);
        lane->active = (warp->active_mask & (1 << i)) != 0;
        lane->gpr[0].u32 = 0; lane->fpr[0].u32 = 0;
        lane->fp_status.float_exception_flags |= float_flag_inexact;
        lane->fp_status.float_rounding_mode = float_round_nearest_even;
        lane->fp_status.default_nan_pattern = 0x7f;
    }
}

int gpgpu_core_threaded_exec_kernel(GPGPUState *s)
{
    uint32_t gd[3]={s->kernel.grid_dim[0],s->kernel.grid_dim[1],s->kernel.grid_dim[2]};
    uint32_t bd[3]={s->kernel.block_dim[0],s->kernel.block_dim[1],s->kernel.block_dim[2]};
    uint32_t ka=s->kernel.kernel_addr, tpb=bd[0]*bd[1]*bd[2];
    for (uint32_t z=0;z<gd[2];z++) for(uint32_t y=0;y<gd[1];y++) for(uint32_t x=0;x<gd[0];x++) {
        uint32_t bid[3]={x,y,z}, bl=z*gd[0]*gd[1]+y*gd[0]+x;
        uint32_t nw=(tpb+GPGPU_WARP_SIZE-1)/GPGPU_WARP_SIZE;
        for(uint32_t w=0;w<nw;w++){
            GPGPUWarp warp; uint32_t tb=w*GPGPU_WARP_SIZE, nt=tpb-tb;
            if(nt>GPGPU_WARP_SIZE)nt=GPGPU_WARP_SIZE;
            gpgpu_core_threaded_init_warp(&warp,ka,tb,bid,nt,w,bl);
            if(gpgpu_core_threaded_exec_warp(s,&warp,100000000)!=0)return -1;
        }
    }
    return 0;
}
