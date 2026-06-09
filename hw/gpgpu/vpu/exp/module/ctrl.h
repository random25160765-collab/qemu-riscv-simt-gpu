/*
 * module/ctrl.h — 控制单元 (分支/跳转/CSR/SIMT 出口/barrier/tex)
 *
 * 聚合: DIV_BR 宏 + rv32i jump/branch + zicsr CSR + GPU 控制指令 (手写)
 */

/* ============================================================
 * 条件分支 — per-lane 求值 + SIMT stack 分歧处理
 * ============================================================ */
#define DIV_BR(cond)                                             \
    do {                                                         \
        int _rs1 = ip[-1].rs1, _rs2 = ip[-1].rs2;                \
        uint32_t _taken = 0;                                     \
        for (int _l = 0; _l < 32; _l++)                          \
            if ((_active >> _l) & 1) {                           \
                uint32_t _a = GPR(_rs1, _l), _b = GPR(_rs2, _l); \
                if (cond) _taken |= (1u << _l);                  \
            }                                                    \
        uint32_t _not_taken = _active & ~_taken;                 \
        for (int _l = 0; _l < 32; _l++) {                        \
            if (_taken & (1u << _l))                             \
                PC(_l) += ip[-1].imm;                            \
            else if (_not_taken & (1u << _l))                    \
                PC(_l) += 4;                                     \
        }                                                        \
        if (perf_enabled) s->stats.total_branches++;             \
        if (_taken && _not_taken) {                              \
            if (_sdepth >= SIMT_STACK_MAX) return -1;            \
            if (perf_enabled) s->stats.simt_diverges++;          \
            _stk[_sdepth].ft_idx = (int32_t)(ip - code);         \
            _stk[_sdepth].mask = _not_taken;                     \
            _sdepth++;                                           \
            _active = _taken;                                    \
            ip = &code[ip[-1].branch_tgt];                       \
        } else if (_taken) {                                     \
            _active = _taken;                                    \
            ip = &code[ip[-1].branch_tgt];                       \
        } else {                                                 \
            _active = _not_taken;                                \
        }                                                        \
        NEXT();                                                  \
    } while (0)

/* 无条件跳转 + 条件分支 */
#include "../inst/rv32i/handles/rv32i_jump.h"
#include "../inst/rv32i/handles/rv32i_branch.h"

/* CSR 寄存器访问 */
#include "../inst/rv32zicsr/handles/zicsr_all.h"
