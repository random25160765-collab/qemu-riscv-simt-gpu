/*
 * gpgpu_core_simd.c — SIMD 解释器（SoA + -O3 自动向量化）
 *
 * 依赖: simd_dispatch.h, simd_predecode.h, simd_handlers.h（解耦模块）
 * 不动 threaded.c / fast.c / gpgpu_core.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "state.h"
#include "gpgpu_core.h"
#include "memory.h"
#include "simd_dispatch.h"
#include "simd_predecode.h"
#include "simd_handlers.h"

static SIMDDecoder simd;
static void __attribute__((constructor)) _simd_ctor(void) { simd_decoder_init(&simd); }

#define G(r,i)  _g [(r)*32+(i)]
#define F(r,i)  _f [(r)*32+(i)]
#define PC(i)   _pc[i]
#define S       for(int i=0;i<32;i++) if(((active)>>i)&1)
#define R       ip[-1]
#define CTRL_BASE 0x80000000
/* per-lane CTRL read */
static inline uint32_t _ctrl_rd(GPGPUState *s, uint32_t addr, int lane) {
    if (addr < CTRL_BASE) return gpu_read(s, addr, 4);
    switch (addr - CTRL_BASE) {
        case 0x00: return s->simt.thread_id[0] + lane;  /* thread_id.x */
        case 0x04: return s->simt.thread_id[1];
        case 0x08: return s->simt.thread_id[2];
        case 0x10: return s->simt.block_id[0];
        case 0x14: return s->simt.block_id[1];
        case 0x18: return s->simt.block_id[2];
        case 0x20: return s->kernel.block_dim[0];
        case 0x24: return s->kernel.block_dim[1];
        case 0x28: return s->kernel.block_dim[2];
        case 0x30: return s->kernel.grid_dim[0];
        case 0x34: return s->kernel.grid_dim[1];
        case 0x38: return s->kernel.grid_dim[2];
        default:   return 0;
    }
}

int gpgpu_core_simd_exec_warp(GPGPUState *s, GPGPUWarp *warp,
                               uint32_t max_cycles, ThOp *code, int tcount) {
    uint32_t _g [GPGPU_NUM_REGS*GPGPU_WARP_SIZE] __attribute__((aligned(32)));
    uint32_t _f [GPGPU_NUM_REGS*GPGPU_WARP_SIZE] __attribute__((aligned(32)));
    uint32_t _pc[GPGPU_WARP_SIZE];
    uint32_t cycles = 0; (void)max_cycles;
    ThOp *ip = code;
    GPGPULane *l = &warp->lanes[0];
    uint32_t active = warp->active_mask;

    for (int i = 0; i < 32; i++) {
        G(0,i) = F(0,i) = 0; PC(i) = warp->lanes[i].pc;
        for (int r = 1; r < GPGPU_NUM_REGS; r++) {
            G(r,i) = warp->lanes[i].gpr[r].u32;
            F(r,i) = warp->lanes[i].fpr[r].u32;
        }
    }

    /* 填充 dispatch 表 */
    if (!simd.dispatch_ready) {
        size_t _di = 0;
        #define X(name, p, t, imm_fn) simd.dispatch[_di++] = &&op_##name;
        INSTRUCTION_LIST
        #undef X
        simd.dispatch_ready = 1;
    }

    /* predecode 存的是 instr_id → 映射到 handler 地址 */
    for (int _i = 0; _i <= tcount; _i++) {
        uint16_t _id = (uint16_t)(uintptr_t)code[_i].handler;
        if (_id > 0 && _id <= simd.op_count)
            code[_i].handler = simd.dispatch[_id - 1];
        else if (_i < tcount)
            code[_i].handler = &&op_illegal;
        else
            code[_i].handler = &&op_done;
    }

    goto *ip++->handler;

    /* ============================================================
     * 分支
     * ============================================================ */
    op_jal:  { int rd=R.rd,imm=R.imm; S{G(rd,i)=PC(i)+4;PC(i)+=imm;} ip=&code[R.branch_tgt]; goto *ip++->handler; }
    op_jalr: { int rd=R.rd,rs1=R.rs1,imm=R.imm; int32_t t=(G(rs1,0)+imm)&~1; S G(rd,i)=PC(i)+4; PC(0)=t; ip=&code[(t-s->kernel.kernel_addr)/4]; goto *ip++->handler; }
    #define BR(n,cond) op_##n: { if(cond){ip=&code[R.branch_tgt];}else{S PC(i)+=4;} goto *ip++->handler; }
    BR(beq,G(R.rs1,0)==G(R.rs2,0)) BR(bne,G(R.rs1,0)!=G(R.rs2,0))
    BR(blt,(int32_t)G(R.rs1,0)<(int32_t)G(R.rs2,0)) BR(bge,(int32_t)G(R.rs1,0)>=(int32_t)G(R.rs2,0))
    BR(bltu,G(R.rs1,0)<G(R.rs2,0)) BR(bgeu,G(R.rs1,0)>=G(R.rs2,0))
    #undef BR

    /* Load */
    op_lb:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=(uint32_t)((int32_t)(gpu_read(s,G(rs1,i)+imm,1)<<24)>>24); S PC(i)+=4; goto *ip++->handler; }
    op_lh:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=(uint32_t)((int32_t)(gpu_read(s,G(rs1,i)+imm,2)<<16)>>16); S PC(i)+=4; goto *ip++->handler; }
    op_lw:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=_ctrl_rd(s,G(rs1,i)+imm,i); S PC(i)+=4; goto *ip++->handler; }
    op_lbu: { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=gpu_read(s,G(rs1,i)+imm,1); S PC(i)+=4; goto *ip++->handler; }
    op_lhu: { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=gpu_read(s,G(rs1,i)+imm,2); S PC(i)+=4; goto *ip++->handler; }

    /* Store */
    op_sb: { int rs1=R.rs1,rs2=R.rs2,imm=R.imm; S gpu_write(s,G(rs1,i)+imm,1,G(rs2,i)); S PC(i)+=4; goto *ip++->handler; }
    op_sh: { int rs1=R.rs1,rs2=R.rs2,imm=R.imm; S gpu_write(s,G(rs1,i)+imm,2,G(rs2,i)); S PC(i)+=4; goto *ip++->handler; }
    op_sw: { int rs1=R.rs1,rs2=R.rs2,imm=R.imm; S gpu_write(s,G(rs1,i)+imm,4,G(rs2,i)); S PC(i)+=4; goto *ip++->handler; }

    /* Upper immediate */
    op_lui:   { int rd=R.rd; uint32_t v=(uint32_t)R.imm; S G(rd,i)=v; S PC(i)+=4; goto *ip++->handler; }
    op_auipc: { int rd=R.rd; uint32_t v=(uint32_t)R.imm; S G(rd,i)=PC(i)+v; S PC(i)+=4; goto *ip++->handler; }

    /* ALU immediate */
    #define AI(n,op) op_##n: { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=G(rs1,i) op (uint32_t)imm; S PC(i)+=4; goto *ip++->handler; }
    AI(addi,+) AI(xori,^) AI(ori,|) AI(andi,&)
    #undef AI
    op_slti:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=((int32_t)G(rs1,i)<imm)?1:0; S PC(i)+=4; goto *ip++->handler; }
    op_sltiu: { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=(G(rs1,i)<(uint32_t)imm)?1:0; S PC(i)+=4; goto *ip++->handler; }
    op_slli:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=G(rs1,i)<<(imm&0x1F); S PC(i)+=4; goto *ip++->handler; }
    op_srli:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=G(rs1,i)>>(imm&0x1F); S PC(i)+=4; goto *ip++->handler; }
    op_srai:  { int rd=R.rd,rs1=R.rs1,imm=R.imm; S G(rd,i)=(uint32_t)((int32_t)G(rs1,i)>>(imm&0x1F)); S PC(i)+=4; goto *ip++->handler; }

    /* ALU register */
    #define AL(n,op) op_##n: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs1,i) op G(rs2,i); S PC(i)+=4; goto *ip++->handler; }
    AL(add,+) AL(sub,-) AL(xor,^) AL(or,|) AL(and,&) AL(mul,*)
    #undef AL
    op_sll:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs1,i)<<(G(rs2,i)&0x1F); S PC(i)+=4; goto *ip++->handler; }
    op_srl:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs1,i)>>(G(rs2,i)&0x1F); S PC(i)+=4; goto *ip++->handler; }
    op_sra:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=(uint32_t)((int32_t)G(rs1,i)>>(G(rs2,i)&0x1F)); S PC(i)+=4; goto *ip++->handler; }
    op_slt:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=((int32_t)G(rs1,i)<(int32_t)G(rs2,i))?1:0; S PC(i)+=4; goto *ip++->handler; }
    op_sltu: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=(G(rs1,i)<G(rs2,i))?1:0; S PC(i)+=4; goto *ip++->handler; }

    /* M extension */
    op_mulh:   { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=(uint32_t)(((int64_t)(int32_t)G(rs1,i)*(int64_t)(int32_t)G(rs2,i))>>32); S PC(i)+=4; goto *ip++->handler; }
    op_mulhsu: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=(uint32_t)(((int64_t)(int32_t)G(rs1,i)*(uint64_t)G(rs2,i))>>32); S PC(i)+=4; goto *ip++->handler; }
    op_mulhu:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=(uint32_t)(((uint64_t)G(rs1,i)*(uint64_t)G(rs2,i))>>32); S PC(i)+=4; goto *ip++->handler; }
    op_div:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs2,i)?(uint32_t)((int32_t)G(rs1,i)/(int32_t)G(rs2,i)):(uint32_t)-1; S PC(i)+=4; goto *ip++->handler; }
    op_divu: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs2,i)?G(rs1,i)/G(rs2,i):0xFFFFFFFF; S PC(i)+=4; goto *ip++->handler; }
    op_rem:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs2,i)?(uint32_t)((int32_t)G(rs1,i)%(int32_t)G(rs2,i)):G(rs1,i); S PC(i)+=4; goto *ip++->handler; }
    op_remu: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=G(rs2,i)?G(rs1,i)%G(rs2,i):G(rs1,i); S PC(i)+=4; goto *ip++->handler; }

    /* CSR — 标量 lane 0 */
    op_csrrw:  { uint16_t c=(uint16_t)R.imm;uint32_t o=_csr_rd(l,c);l->gpr[R.rd].u32=o;_csr_wr(l,c,l->gpr[R.rs1].u32);l->pc+=4; goto *ip++->handler; }
    op_csrrs:  { uint16_t c=(uint16_t)R.imm;uint32_t o=_csr_rd(l,c);l->gpr[R.rd].u32=o;if(R.rs1!=0)_csr_wr(l,c,o|l->gpr[R.rs1].u32);l->pc+=4; goto *ip++->handler; }
    op_csrrc:  { uint16_t c=(uint16_t)R.imm;uint32_t o=_csr_rd(l,c);l->gpr[R.rd].u32=o;if(R.rs1!=0)_csr_wr(l,c,o&~l->gpr[R.rs1].u32);l->pc+=4; goto *ip++->handler; }
    op_csrrwi: { uint16_t c=(uint16_t)R.imm;l->gpr[R.rd].u32=_csr_rd(l,c);_csr_wr(l,c,(uint32_t)R.rs1);l->pc+=4; goto *ip++->handler; }
    op_csrrsi: { uint16_t c=(uint16_t)R.imm;uint32_t o=_csr_rd(l,c);l->gpr[R.rd].u32=o;if(R.rs1!=0)_csr_wr(l,c,o|(uint32_t)R.rs1);l->pc+=4; goto *ip++->handler; }
    op_csrrci: { uint16_t c=(uint16_t)R.imm;uint32_t o=_csr_rd(l,c);l->gpr[R.rd].u32=o;if(R.rs1!=0)_csr_wr(l,c,o&~(uint32_t)R.rs1);l->pc+=4; goto *ip++->handler; }

    /* ebreak/done — SoA→AoS */
    op_ebreak: {
        for(int i=0;i<32;i++){warp->lanes[i].pc=PC(i);
            for(int r=0;r<GPGPU_NUM_REGS;r++){warp->lanes[i].gpr[r].u32=G(r,i);warp->lanes[i].fpr[r].u32=F(r,i);}}
        return 0;
    }
    op_done: { goto op_ebreak; }

    /* ============================================================
     * FP — 硬件浮点 + SoA，-O3 自动向量化
     * 用 union 做 type punning，不触发 strict-aliasing
     * ============================================================ */
    #define VRAM_RD(addr) (*(uint32_t*)(s->vram_ptr + (addr)))
    /* FP 数组指针 — SoA layout: 32 regs × 32 lanes, 每 lane 连续 */
    float *fp = (float*)_f; (void)fp;
    #define FR(r,i) ((float*)_f)[(r)*32+(i)]  /* 直接 cast，不依赖局部变量 */

    /* FLW/FSW — VRAM fast path */
    op_flw: { int rd=R.rd,rs1=R.rs1,imm=R.imm;
        S {uint32_t a=G(rs1,i)+imm; F(rd,i)=VRAM_RD(a);} S PC(i)+=4; goto *ip++->handler; }
    op_fsw: { int rs1=R.rs1,rs2=R.rs2,imm=R.imm;
        S {uint32_t a=G(rs1,i)+imm; VRAM_RD(a)=F(rs2,i);} S PC(i)+=4; goto *ip++->handler; }

    /* basic FP — direct float array access, -O3 vectorizes to vmulps/vaddps */
    #define HF(n,op) op_##n: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; \
        for(int i=0;i<32;i++) FR(rd,i)=FR(rs1,i) op FR(rs2,i); \
        S PC(i)+=4; goto *ip++->handler; }
    HF(fmul_s,*) HF(fadd_s,+) HF(fsub_s,-) HF(fdiv_s,/)
    #undef HF

    /* fmadd — FMA */
    op_fmadd_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2,rs3=R.rs3;
        S FR(rd,i)=FR(rs1,i)*FR(rs2,i)+FR(rs3,i);
        S PC(i)+=4; goto *ip++->handler; }

    /* fsqrt — per-lane libm call */
    op_fsqrt_s: { int rd=R.rd,rs1=R.rs1;
        S FR(rd,i)=sqrtf(FR(rs1,i)); S PC(i)+=4; goto *ip++->handler; }

    /* sign inject — bit ops */
    op_fsgnj_s:  { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S F(rd,i)=(F(rs1,i)&~0x80000000)|(F(rs2,i)&0x80000000); S PC(i)+=4; goto *ip++->handler; }
    op_fsgnjn_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S F(rd,i)=(F(rs1,i)&~0x80000000)|((~F(rs2,i))&0x80000000); S PC(i)+=4; goto *ip++->handler; }
    op_fsgnjx_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S F(rd,i)=F(rs1,i)^(F(rs2,i)&0x80000000); S PC(i)+=4; goto *ip++->handler; }
    op_fmin_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2;
        S FR(rd,i)=FR(rs1,i)<FR(rs2,i)?FR(rs1,i):FR(rs2,i); S PC(i)+=4; goto *ip++->handler; }
    op_fmax_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2;
        S FR(rd,i)=FR(rs1,i)>FR(rs2,i)?FR(rs1,i):FR(rs2,i); S PC(i)+=4; goto *ip++->handler; }

    /* FMV — 位拷贝 */
    op_fmv_w_x:  { int rd=R.rd,rs1=R.rs1; S F(rd,i)=G(rs1,i); S PC(i)+=4; goto *ip++->handler; }
    op_fmv_x_w:  { int rd=R.rd,rs1=R.rs1; S G(rd,i)=F(rs1,i); S PC(i)+=4; goto *ip++->handler; }
    /* FCVT int→float */
    op_fcvt_s_w: { int rd=R.rd,rs1=R.rs1; S FR(rd,i)=(float)(int32_t)G(rs1,i); S PC(i)+=4; goto *ip++->handler; }
    op_fcvt_s_wu:{ int rd=R.rd,rs1=R.rs1; S FR(rd,i)=(float)G(rs1,i); S PC(i)+=4; goto *ip++->handler; }
    /* FCVT float→int */
    op_fcvt_w_s: { int rd=R.rd,rs1=R.rs1;
        S {float f=FR(rs1,i);uint32_t raw=F(rs1,i);
            G(rd,i)=(uint32_t)(int32_t)(((raw>>23)&0xFF)==0xFF&&(raw&0x7FFFFF)?0x7FFFFFFF:(f>2.147e9f?0x7FFFFFFF:(f<-2.147e9f?0x80000000:(int32_t)f)));}
        S PC(i)+=4; goto *ip++->handler; }
    op_fcvt_wu_s:{ int rd=R.rd,rs1=R.rs1;
        S {float f=FR(rs1,i);uint32_t raw=F(rs1,i);
            G(rd,i)=((raw>>23)&0xFF)==0xFF&&(raw&0x7FFFFF)?0xFFFFFFFF:(f<0?0:(f>4.294e9f?0xFFFFFFFF:(uint32_t)f));}
        S PC(i)+=4; goto *ip++->handler; }

    /* compare */
    op_feq_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=F(rs1,i)==F(rs2,i)?1:0; S PC(i)+=4; goto *ip++->handler; }
    op_flt_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=FR(rs1,i)<FR(rs2,i)?1:0; S PC(i)+=4; goto *ip++->handler; }
    op_fle_s: { int rd=R.rd,rs1=R.rs1,rs2=R.rs2; S G(rd,i)=FR(rs1,i)<=FR(rs2,i)?1:0; S PC(i)+=4; goto *ip++->handler; }

    /* fclass */
    op_fclass_s:{ int rd=R.rd,rs1=R.rs1; S {uint32_t b=F(rs1,i),e=(b>>23)&0xFF,m=b&0x7FFFFF,s=(b>>31)&1;int r=0;
        if(e==0xFF)r=m?(s?(1<<9):(1<<8)):(s?(1<<0):(1<<7));else if(e==0)r=m?(s?(1<<2):(1<<5)):(s?(1<<3):(1<<4));else r=s?(1<<1):(1<<6);
        G(rd,i)=r;} S PC(i)+=4; goto *ip++->handler; }

    /* LP float — scalar lane 0 (rare) */
    #define LPS(n,body) op_##n: { body; l->pc+=4; goto *ip++->handler; }
    LPS(fcvt_s_bf16,l->fpr[R.rd].u32=bf16_to_f32(l->fpr[R.rs1].bf16))
    LPS(fcvt_bf16_s,l->fpr[R.rd].bf16=f32_to_bf16(l->fpr[R.rs1].u32))
    LPS(fcvt_s_e4m3,l->fpr[R.rd].u32=e4m3_to_f32(l->fpr[R.rs1].e4m3))
    LPS(fcvt_e4m3_s,l->fpr[R.rd].e4m3=f32_to_e4m3(l->fpr[R.rs1].u32))
    LPS(fcvt_s_e5m2,l->fpr[R.rd].u32=e5m2_to_f32(l->fpr[R.rs1].e5m2))
    LPS(fcvt_e5m2_s,l->fpr[R.rd].e5m2=f32_to_e5m2(l->fpr[R.rs1].u32))
    LPS(fcvt_s_e2m1,l->fpr[R.rd].u32=e2m1_to_f32(l->fpr[R.rs1].e2m1))
    LPS(fcvt_e2m1_s,l->fpr[R.rd].e2m1=f32_to_e2m1(l->fpr[R.rs1].u32))

    /* 科学计算 — 硬件 libm per-lane */
    #define SCI_SOA(n,expr) op_##n: { int rd=R.rd,rs1=R.rs1; \
        S {float v=FR(rs1,i); FR(rd,i)=expr;} \
        S PC(i)+=4; goto *ip++->handler; }
    SCI_SOA(fexp_s, expf(v))
    SCI_SOA(fln_s, logf(v))
    SCI_SOA(frcp_s, 1.0f/v)
    SCI_SOA(frsqrt_s, 1.0f/sqrtf(v))
    SCI_SOA(ftanh_s, tanhf(v))
    SCI_SOA(fsigmoid_s, 1.0f/(1.0f+expf(-v)))
    SCI_SOA(fsin_s, sinf(v))
    SCI_SOA(fcos_s, cosf(v))
    #undef SCI_SOA
    #undef F32
    #undef VRAM_RD

    /* fallback FP/SYS — FMA variants */
    #define FPFB(n,body) op_##n: { FP_SYNC(l); body; FP_BACK(l); l->pc+=4; goto *ip++->handler; }
    FPFB(fmsub_s, l->fpr[R.rd].u32=float32_muladd(l->fpr[R.rs1].u32,l->fpr[R.rs2].u32,l->fpr[R.rs3].u32,float_muladd_negate_c,&l->fp_status))
    FPFB(fnmsub_s,l->fpr[R.rd].u32=float32_muladd(l->fpr[R.rs1].u32,l->fpr[R.rs2].u32,l->fpr[R.rs3].u32,float_muladd_negate_product,&l->fp_status))
    FPFB(fnmadd_s,l->fpr[R.rd].u32=float32_muladd(l->fpr[R.rs1].u32,l->fpr[R.rs2].u32,l->fpr[R.rs3].u32,float_muladd_negate_result,&l->fp_status))
    #undef FPFB

    op_illegal: {
        for(int i=0;i<32;i++){warp->lanes[i].pc=PC(i);
            for(int r=0;r<GPGPU_NUM_REGS;r++){warp->lanes[i].gpr[r].u32=G(r,i);warp->lanes[i].fpr[r].u32=F(r,i);}}
        return -1;
    }
}

void gpgpu_core_simd_init_warp(GPGPUWarp *warp, uint32_t pc,
    uint32_t tid_base, const uint32_t block_id[3],
    uint32_t num_threads, uint32_t warp_id, uint32_t blk_linear) {
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

int gpgpu_core_simd_exec_kernel(GPGPUState *s) {
    uint32_t gd[3]={s->kernel.grid_dim[0],s->kernel.grid_dim[1],s->kernel.grid_dim[2]};
    uint32_t bd[3]={s->kernel.block_dim[0],s->kernel.block_dim[1],s->kernel.block_dim[2]};
    uint32_t ka=s->kernel.kernel_addr, tpb=bd[0]*bd[1]*bd[2];
    for (uint32_t z=0;z<gd[2];z++) for(uint32_t y=0;y<gd[1];y++) for(uint32_t x=0;x<gd[0];x++) {
        uint32_t bid[3]={x,y,z}, bl=z*gd[0]*gd[1]+y*gd[0]+x;
        uint32_t nw=(tpb+GPGPU_WARP_SIZE-1)/GPGPU_WARP_SIZE;
        for(uint32_t w=0;w<nw;w++){
            GPGPUWarp warp; uint32_t tb=w*GPGPU_WARP_SIZE, nt=tpb-tb;
            if(nt>GPGPU_WARP_SIZE)nt=GPGPU_WARP_SIZE;
            gpgpu_core_simd_init_warp(&warp,ka,tb,bid,nt,w,bl);
            uint32_t ks=4096; int tc=0;
            ThOp *code = simd_predecode(&simd, s, ka, ks, &tc);
            int ret = gpgpu_core_simd_exec_warp(s, &warp, 100000000, code, tc);
            free(code);
            if (ret != 0) return -1;
        }
    }
    return 0;
}
