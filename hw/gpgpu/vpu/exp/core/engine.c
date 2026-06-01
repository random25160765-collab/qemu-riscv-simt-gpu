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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "state.h"
#include "gpgpu_core.h"
#include "engine.h"
#include "lpfp.h"
#include "memory.h"
#include "simd_dispatch.h"    /* INSTRUCTION_LIST (唯一来源) */
#include "simd_predecode.h"   /* ThOp */

/* ============================================================
 * SoA 数据布局宏
 * ============================================================ */
#define GPR(reg, lane)  gpr[(reg) * 32 + (lane)]
#define FPR(reg, lane)  fpr[(reg) * 32 + (lane)]
#define PC(lane)        _pc[lane]
#define FOR_EACH_LANE   for (int _li = 0; _li < 32; _li++) if ((_active >> _li) & 1)

/* float 视图 (SoA)，供 -O3 自动向量化 */
#define FR(reg, lane)   ((float *)fpr)[(reg) * 32 + (lane)]

/* ============================================================
 * CTRL 寄存器 per-lane 读取
 * ============================================================ */
static inline uint32_t ctrl_read(const EngineContext *ctx, uint32_t addr, int lane)
{
    if (addr < 0x80000000)
        return gpu_read(ctx->s, addr, 4);

    switch (addr - 0x80000000) {
        case 0x00: return ctx->thread_id_base + lane;  /* thread_id.x */
        case 0x04: return 0;                            /* thread_id.y */
        case 0x08: return 0;                            /* thread_id.z */
        case 0x10: return ctx->block_id[0];
        case 0x14: return ctx->block_id[1];
        case 0x18: return ctx->block_id[2];
        case 0x20: return ctx->s->kernel.block_dim[0];
        case 0x24: return ctx->s->kernel.block_dim[1];
        case 0x28: return ctx->s->kernel.block_dim[2];
        case 0x30: return ctx->s->kernel.grid_dim[0];
        case 0x34: return ctx->s->kernel.grid_dim[1];
        case 0x38: return ctx->s->kernel.grid_dim[2];
        default:   return 0;
    }
}

/* ============================================================
 * Dispatch 表 + Handler 解析
 * 必须在 engine_exec 内部，因为 &&op_* 标签是函数作用域
 * ============================================================ */
#define NUM_OF_INST 300
static void *dispatch[NUM_OF_INST];
static int dispatch_ready = 0;

void engine_resolve_handlers(ThOp *code, int tcount)
{
    (void)code;
    (void)tcount;
    /* 实际解析在 engine_exec 内部进行 (lazy init + resolve) */
}

/* ============================================================
 * 引擎入口
 * ============================================================ */
int engine_exec(ThOp *code, int tcount, const EngineContext *ctx,
                uint32_t gpr[32 * 32], uint32_t fpr[32 * 32], uint32_t _pc[32],
                uint32_t _mhartid[32], uint32_t _fcsr[32])
{
    ThOp *ip;
    uint32_t _active = ctx->active;
    GPGPUState *s = ctx->s;
    (void)tcount;

    /* 短别名供 CSR/LP 使用 */
    #define MHARTID(l) _mhartid[l]
    #define FCSR(l)    _fcsr[l]

    /* ============================================================
     * SIMT stack — per-lane 控制流 divergence/reconvergence
     *
     * 栈帧: { fallthrough_idx, mask }
     *   分歧时 push not_taken 路径, 先执行 taken 路径。
     *   taken 路径所有 lane ebreak 后 pop 栈, 恢复 not_taken 路径。
     *   寄存器是 SoA per-lane 的, 不活跃 lane 的值自然保留。
     * ============================================================ */
    #define SIMT_STACK_MAX 32
    struct { int32_t ft_idx; uint32_t mask; } _stk[SIMT_STACK_MAX];
    int _sdepth = 0;

    /* === lazy init: 填充 dispatch 表 + 解析 handler === */
    if (!dispatch_ready) {
        size_t di = 0;
        #define X(name, pattern, op_type, imm_fn) dispatch[di++] = &&op_##name;
        INSTRUCTION_LIST
        #undef X
        dispatch_ready = 1;
    }

    /* 解析 handler（ThOp 缓存后只做一次） */
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
    s->inst_count += __builtin_popcount(_active); goto *ip++->handler;

    /* ============================================================
     * 无条件跳转 (所有 lane 统一, 不发散)
     * ============================================================ */
    op_jal: {
        int rd = ip[-1].rd, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = PC(_li) + 4; PC(_li) += imm; }
        ip = &code[ip[-1].branch_tgt];
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_jalr: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        int32_t target = (int32_t)(GPR(rs1, 0) + imm) & ~1;
        FOR_EACH_LANE { GPR(rd, _li) = PC(_li) + 4; PC(_li) = (uint32_t)target; }
        ip = &code[(target - s->kernel.kernel_addr) / 4];
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
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
    #define DIV_BR(cond) do { \
        int _rs1 = ip[-1].rs1, _rs2 = ip[-1].rs2; \
        uint32_t _taken = 0; \
        for (int _l = 0; _l < 32; _l++) \
            if ((_active >> _l) & 1) { \
                uint32_t _a = GPR(_rs1, _l), _b = GPR(_rs2, _l); \
                if (cond) _taken |= (1u << _l); \
            } \
        uint32_t _not_taken = _active & ~_taken; \
        /* PC 更新: 必须在改 _active 之前, 对两组分别处理 */ \
        for (int _l = 0; _l < 32; _l++) { \
            if (_taken & (1u << _l)) \
                PC(_l) += ip[-1].imm; \
            else if (_not_taken & (1u << _l)) \
                PC(_l) += 4; \
        } \
        if (_taken && _not_taken) { \
            /* 分歧: push not_taken, 先跑 taken */ \
            _stk[_sdepth].ft_idx = (int32_t)(ip - code); \
            _stk[_sdepth].mask   = _not_taken; \
            _sdepth++; \
            _active = _taken; \
            ip = &code[ip[-1].branch_tgt]; \
        } else if (_taken) { \
            _active = _taken; \
            ip = &code[ip[-1].branch_tgt]; \
        } else { \
            _active = _not_taken; \
        } \
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler; \
    } while(0)

    op_beq:  DIV_BR(_a == _b);
    op_bne:  DIV_BR(_a != _b);
    op_blt:  DIV_BR((int32_t)_a <  (int32_t)_b);
    op_bge:  DIV_BR((int32_t)_a >= (int32_t)_b);
    op_bltu: DIV_BR(_a <  _b);
    op_bgeu: DIV_BR(_a >= _b);
    #undef DIV_BR

    /* ============================================================
     * Load — per-lane 地址，per-lane CTRL 读
     * ============================================================ */
    op_lb: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            uint8_t v = (uint8_t)gpu_read(s, a, 1);
            GPR(rd, _li) = (uint32_t)((int32_t)(v << 24) >> 24);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_lh: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            uint16_t v = (uint16_t)gpu_read(s, a, 2);
            GPR(rd, _li) = (uint32_t)((int32_t)(v << 16) >> 16);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_lw: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            GPR(rd, _li) = ctrl_read(ctx, a, _li);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_lbu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            GPR(rd, _li) = (uint32_t)gpu_read(s, a, 1);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_lhu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            GPR(rd, _li) = (uint32_t)gpu_read(s, a, 2);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* Store */
    op_sb: {
        int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
        FOR_EACH_LANE { gpu_write(s, GPR(rs1, _li) + imm, 1, GPR(rs2, _li)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_sh: {
        int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
        FOR_EACH_LANE { gpu_write(s, GPR(rs1, _li) + imm, 2, GPR(rs2, _li)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_sw: {
        int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
        FOR_EACH_LANE { gpu_write(s, GPR(rs1, _li) + imm, 4, GPR(rs2, _li)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * Upper immediate
     * ============================================================ */
    op_lui: {
        int rd = ip[-1].rd; uint32_t v = (uint32_t)ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = v; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_auipc: {
        int rd = ip[-1].rd; uint32_t v = (uint32_t)ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = PC(_li) + v; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * ALU immediate
     * ============================================================ */
    #define ALUI(name, op) \
    op_##name: { \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm; \
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) op (uint32_t)imm; PC(_li) += 4; } \
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler; \
    }
    ALUI(addi, +) ALUI(xori, ^) ALUI(ori, |) ALUI(andi, &)
    #undef ALUI

    op_slti: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = ((int32_t)GPR(rs1, _li) < imm) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_sltiu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = (GPR(rs1, _li) < (uint32_t)imm) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_slli: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) << (imm & 0x1F); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_srli: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) >> (imm & 0x1F); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_srai: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE { GPR(rd, _li) = (uint32_t)((int32_t)GPR(rs1, _li) >> (imm & 0x1F)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * ALU register
     * ============================================================ */
    #define ALUR(name, op) \
    op_##name: { \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) op GPR(rs2, _li); PC(_li) += 4; } \
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler; \
    }
    ALUR(add, +) ALUR(sub, -) ALUR(xor, ^) ALUR(or, |) ALUR(and, &) ALUR(mul, *)
    #undef ALUR

    op_sll: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) << (GPR(rs2, _li) & 0x1F); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_srl: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = GPR(rs1, _li) >> (GPR(rs2, _li) & 0x1F); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_sra: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = (uint32_t)((int32_t)GPR(rs1, _li) >> (GPR(rs2, _li) & 0x1F)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_slt: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = ((int32_t)GPR(rs1, _li) < (int32_t)GPR(rs2, _li)) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_sltu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = (GPR(rs1, _li) < GPR(rs2, _li)) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * RV32M multiply/divide
     * ============================================================ */
    op_mulh: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = (uint32_t)(((int64_t)(int32_t)GPR(rs1, _li) * (int64_t)(int32_t)GPR(rs2, _li)) >> 32); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_mulhsu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = (uint32_t)(((int64_t)(int32_t)GPR(rs1, _li) * (uint64_t)GPR(rs2, _li)) >> 32); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_mulhu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = (uint32_t)(((uint64_t)GPR(rs1, _li) * (uint64_t)GPR(rs2, _li)) >> 32); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_div: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE {
            int32_t a = (int32_t)GPR(rs1, _li), b = (int32_t)GPR(rs2, _li);
            GPR(rd, _li) = b ? (uint32_t)(a / b) : (uint32_t)-1; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_divu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li), b = GPR(rs2, _li);
            GPR(rd, _li) = b ? a / b : 0xFFFFFFFF; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_rem: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE {
            int32_t a = (int32_t)GPR(rs1, _li), b = (int32_t)GPR(rs2, _li);
            GPR(rd, _li) = b ? (uint32_t)(a % b) : GPR(rs1, _li); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_remu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li), b = GPR(rs2, _li);
            GPR(rd, _li) = b ? a % b : a; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
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
     * 但由于 GPGPULane 有 fp_status (softfloat 状态) 且 FP 指令需要它，
     * 引擎实际上需要每 lane 的 fp_status。这很复杂。
     *
     * 当前简化：CSR 只操作 lane 0，依赖调度器保留的 warp->lanes[0]。
     * 引擎需要额外的 lane 状态指针来做 CSR 操作。
     *
     * FIXME: 将 lane 元数据 (mhartid, fcsr, fp_status) 也 SoA 化。
     */
    /* 暂时通过 ctx->s 和 warp lane 0 的引用做 CSR，过渡方案 */

    /* ============================================================
     * ebreak / done
     * ============================================================ */
    /* ============================================================
     * ebreak / done — SIMT stack 感知出口
     *
     * 所有活跃 lane 退出 → _active=0。
     * 栈非空 → pop 恢复之前 push 的 not_taken 路径，继续执行。
     * 栈空   → 整个 warp 完成, return 0。
     * ============================================================ */
    op_ebreak: {
        _active = 0;
        if (_sdepth > 0) {
            _sdepth--;
            _active = _stk[_sdepth].mask;
            ip = &code[_stk[_sdepth].ft_idx];
            s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
        }
        return 0;
    }
    op_done: {
        _active = 0;
        if (_sdepth > 0) {
            _sdepth--;
            _active = _stk[_sdepth].mask;
            ip = &code[_stk[_sdepth].ft_idx];
            s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
        }
        return 0;
    }

    /* ============================================================
     * FP Load/Store — VRAM fast path
     * ============================================================ */
    op_flw: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            FPR(rd, _li) = *(uint32_t *)(s->vram_ptr + a);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fsw: {
        int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
        FOR_EACH_LANE {
            uint32_t a = GPR(rs1, _li) + imm;
            *(uint32_t *)(s->vram_ptr + a) = FPR(rs2, _li);
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * FP 基础算术 — 硬件 float，-O3 自动向量化为 vmulps/vaddps
     * ============================================================ */
    #define FPBIN(name, op) \
    op_##name: { \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2; \
        FOR_EACH_LANE { FR(rd, _li) = FR(rs1, _li) op FR(rs2, _li); } \
        FOR_EACH_LANE PC(_li) += 4; \
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler; \
    }
    FPBIN(fadd_s, +) FPBIN(fsub_s, -) FPBIN(fmul_s, *) FPBIN(fdiv_s, /)
    #undef FPBIN

    /* FMA — SoA 展开 (之前 simd 缺失的变体也补全) */
    op_fmadd_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;
        FOR_EACH_LANE { FR(rd, _li) = FR(rs1, _li) * FR(rs2, _li) + FR(rs3, _li); }
        FOR_EACH_LANE PC(_li) += 4;
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fmsub_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;
        FOR_EACH_LANE { FR(rd, _li) = FR(rs1, _li) * FR(rs2, _li) - FR(rs3, _li); }
        FOR_EACH_LANE PC(_li) += 4;
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fnmsub_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;
        FOR_EACH_LANE { FR(rd, _li) = -(FR(rs1, _li) * FR(rs2, _li) - FR(rs3, _li)); }
        FOR_EACH_LANE PC(_li) += 4;
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fnmadd_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;
        FOR_EACH_LANE { FR(rd, _li) = -(FR(rs1, _li) * FR(rs2, _li) + FR(rs3, _li)); }
        FOR_EACH_LANE PC(_li) += 4;
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* fsqrt — per-lane libm */
    op_fsqrt_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE { FR(rd, _li) = sqrtf(FR(rs1, _li)); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* 符号注入 — 位操作 */
    op_fsgnj_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { FPR(rd, _li) = (FPR(rs1, _li) & ~0x80000000) | (FPR(rs2, _li) & 0x80000000); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fsgnjn_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { FPR(rd, _li) = (FPR(rs1, _li) & ~0x80000000) | ((~FPR(rs2, _li)) & 0x80000000); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fsgnjx_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { FPR(rd, _li) = FPR(rs1, _li) ^ (FPR(rs2, _li) & 0x80000000); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* 最值 */
    op_fmin_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { FR(rd, _li) = FR(rs1, _li) < FR(rs2, _li) ? FR(rs1, _li) : FR(rs2, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fmax_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { FR(rd, _li) = FR(rs1, _li) > FR(rs2, _li) ? FR(rs1, _li) : FR(rs2, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * FP 比较 (整数结果写 gpr)
     * ============================================================ */
    op_feq_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = FR(rs1, _li) == FR(rs2, _li) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_flt_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = FR(rs1, _li) < FR(rs2, _li) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fle_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE { GPR(rd, _li) = FR(rs1, _li) <= FR(rs2, _li) ? 1 : 0; PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * FP 转换
     * ============================================================ */
    op_fcvt_s_w: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE { FR(rd, _li) = (float)(int32_t)GPR(rs1, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_s_wu: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE { FR(rd, _li) = (float)GPR(rs1, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_w_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            float f = FR(rs1, _li); uint32_t raw = FPR(rs1, _li);
            GPR(rd, _li) = (uint32_t)(int32_t)(
                ((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF) ? 0x7FFFFFFF :
                (f > 2.147e9f ? 0x7FFFFFFF : (f < -2.147e9f ? (int32_t)0x80000000 : (int32_t)f)));
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_wu_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            float f = FR(rs1, _li); uint32_t raw = FPR(rs1, _li);
            GPR(rd, _li) = ((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF) ? 0xFFFFFFFF :
                (f < 0.0f ? 0 : (f > 4.294e9f ? 0xFFFFFFFF : (uint32_t)f));
            PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* 数据移动 */
    op_fmv_w_x: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE { FPR(rd, _li) = GPR(rs1, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fmv_x_w: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE { GPR(rd, _li) = FPR(rs1, _li); PC(_li) += 4; }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* fclass */
    op_fclass_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t b = FPR(rs1, _li), e = (b >> 23) & 0xFF, m = b & 0x7FFFFF, sgn = (b >> 31) & 1;
            int r = 0;
            if (e == 0xFF) r = m ? (sgn ? (1 << 9) : (1 << 8)) : (sgn ? (1 << 0) : (1 << 7));
            else if (e == 0) r = m ? (sgn ? (1 << 2) : (1 << 5)) : (sgn ? (1 << 3) : (1 << 4));
            else r = sgn ? (1 << 1) : (1 << 6);
            GPR(rd, _li) = (uint32_t)r; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * 科学计算 — SoA + libm (每次迭代调用数学库)
     * ============================================================ */
    #define SCI(name, expr) \
    op_##name: { \
        int rd = ip[-1].rd, rs1 = ip[-1].rs1; \
        FOR_EACH_LANE { float v = FR(rs1, _li); FR(rd, _li) = expr; PC(_li) += 4; } \
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler; \
    }
    SCI(fexp_s,     expf(v))
    SCI(fln_s,      logf(v))
    SCI(frcp_s,     1.0f / v)
    SCI(frsqrt_s,   1.0f / sqrtf(v))
    SCI(ftanh_s,    tanhf(v))
    SCI(fsigmoid_s, 1.0f / (1.0f + expf(-v)))
    SCI(fsin_s,     sinf(v))
    SCI(fcos_s,     cosf(v))
    #undef SCI

    /* ============================================================
     * CSR — per-lane 标量 (每个 lane 有独立的 mhartid/fcsr)
     * imm 即 CSR 地址 (12-bit)
     * ============================================================ */
    op_csrrw: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t old = 0;
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         MHARTID(_li) = GPR(rs1, _li); break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    FCSR(_li) = (FCSR(_li) & ~0x1F) | (GPR(rs1, _li) & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; FCSR(_li) = (FCSR(_li) & ~0xE0) | ((GPR(rs1, _li) & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            FCSR(_li) = GPR(rs1, _li); break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_csrrs: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t old = 0, mask = GPR(rs1, _li);
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         if (rs1) MHARTID(_li) |= mask; break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    if (rs1) FCSR(_li) |= (mask & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; if (rs1) FCSR(_li) |= ((mask & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            if (rs1) FCSR(_li) |= mask; break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_csrrc: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t old = 0, mask = GPR(rs1, _li);
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         if (rs1) MHARTID(_li) &= ~mask; break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    if (rs1) FCSR(_li) &= ~(mask & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; if (rs1) FCSR(_li) &= ~((mask & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            if (rs1) FCSR(_li) &= ~mask; break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_csrrwi: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd; uint32_t zimm = (uint32_t)ip[-1].rs1; /* rs1 字段 = uimm5 */
        FOR_EACH_LANE {
            uint32_t old = 0;
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         MHARTID(_li) = zimm; break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    FCSR(_li) = (FCSR(_li) & ~0x1F) | (zimm & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; FCSR(_li) = (FCSR(_li) & ~0xE0) | ((zimm & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            FCSR(_li) = zimm; break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_csrrsi: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd; uint32_t zimm = (uint32_t)ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t old = 0;
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         if (ip[-1].rs1) MHARTID(_li) |= zimm; break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    if (ip[-1].rs1) FCSR(_li) |= (zimm & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; if (ip[-1].rs1) FCSR(_li) |= ((zimm & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            if (ip[-1].rs1) FCSR(_li) |= zimm; break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_csrrci: {
        uint16_t csr = (uint16_t)ip[-1].imm;
        int rd = ip[-1].rd; uint32_t zimm = (uint32_t)ip[-1].rs1;
        FOR_EACH_LANE {
            uint32_t old = 0;
            switch (csr) {
                case CSR_MHARTID: old = MHARTID(_li);         if (ip[-1].rs1) MHARTID(_li) &= ~zimm; break;
                case CSR_FFLAGS:  old = FCSR(_li) & 0x1F;    if (ip[-1].rs1) FCSR(_li) &= ~(zimm & 0x1F); break;
                case CSR_FRM:     old = (FCSR(_li) >> 5) & 7; if (ip[-1].rs1) FCSR(_li) &= ~((zimm & 7) << 5); break;
                case CSR_FCSR:    old = FCSR(_li);            if (ip[-1].rs1) FCSR(_li) &= ~zimm; break;
            }
            GPR(rd, _li) = old; PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * LP 浮点转换 — per-lane (位域操作, lpfp 接受值类型)
     * bf16 在 fpr 的 bits[31:16], e4m3/e5m2/e2m1 在 bits[31:24]
     * ============================================================ */
    op_fcvt_s_bf16: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint16_t bf = (uint16_t)(FPR(rs1, _li) >> 16);
            FPR(rd, _li) = bf16_to_f32(bf); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_bf16_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint16_t bf = f32_to_bf16(FPR(rs1, _li));
            FPR(rd, _li) = (FPR(rd, _li) & 0xFFFF) | ((uint32_t)bf << 16); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_s_e4m3: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e4 = (uint8_t)(FPR(rs1, _li) >> 24);
            FPR(rd, _li) = e4m3_to_f32(e4); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_e4m3_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e4 = f32_to_e4m3(FPR(rs1, _li));
            FPR(rd, _li) = (FPR(rd, _li) & 0xFFFFFF) | ((uint32_t)e4 << 24); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_s_e5m2: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e5 = (uint8_t)(FPR(rs1, _li) >> 24);
            FPR(rd, _li) = e5m2_to_f32(e5); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_e5m2_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e5 = f32_to_e5m2(FPR(rs1, _li));
            FPR(rd, _li) = (FPR(rd, _li) & 0xFFFFFF) | ((uint32_t)e5 << 24); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_s_e2m1: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e2 = (uint8_t)(FPR(rs1, _li) >> 24);
            FPR(rd, _li) = e2m1_to_f32(e2); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }
    op_fcvt_e2m1_s: {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE {
            uint8_t e2 = f32_to_e2m1(FPR(rs1, _li));
            FPR(rd, _li) = (FPR(rd, _li) & 0xFFFFFF) | ((uint32_t)e2 << 24); PC(_li) += 4;
        }
        s->inst_count += __builtin_popcount(_active); goto *ip++->handler;
    }

    /* ============================================================
     * 错误出口
     * ============================================================ */
    op_illegal: { return -1; }
}
