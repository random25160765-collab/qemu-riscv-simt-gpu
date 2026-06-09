/*
 * dispatch_mod.h — 模块化 dispatch 表
 *
 * 与 dispatch.h 的区别:
 *   INSTRUCTION_LIST 由 inst/scripts/gen_dispatch.py 从所有 ISA YAML spec 聚合生成,
 *   不再手工维护。原始 dispatch.h 保留作为参考。
 *
 * 依赖生成文件:
 *   ../inst/dispatch_list.h   — DISP_ enum + INSTRUCTION_LIST 宏 + NUM_OF_INST
 *   ../inst/dispatch_rebind.h — rvv_dispatch_rebind(dispatch, sew)
 */
#ifndef SIMD_DISPATCH_MOD_H
#define SIMD_DISPATCH_MOD_H

typedef enum {
    TYPE_R,
    TYPE_I,
    TYPE_U,
    TYPE_S,
    TYPE_J,
    TYPE_B,
    TYPE_CSR,
    TYPE_FR,
    TYPE_FI,
    TYPE_FS,
    TYPE_F4
} inst_type_t;

#include "inst.h"
#include "utils.h"
#include "decode_trie.h"
#include "state.h"

/* ============================================================
 * INSTRUCTION_LIST — 由 gen_dispatch.py 从 ISA spec 聚合生成
 * ============================================================ */
#include "../inst/dispatch_list.h"
/* 提供: DISP_ enum + INSTRUCTION_LIST 宏 + NUM_OF_INST */

#include "../inst/dispatch_rebind.h"
/* 提供: rvv_dispatch_rebind(dispatch, sew) — vsetvli 热替换 */

static inline int32_t imm0(uint32_t i)
{
    (void)i;
    return 0;
}

/* ============================================================
 * Dispatch 表 + trie（每个 simd 实例独立一份）
 * ============================================================ */
#define SIMD_NUM_INST NUM_OF_INST

typedef struct {
    void *dispatch[SIMD_NUM_INST];
    int dispatch_ready;
    DecodeTrie trie;
    opcode_entry_t op_table[SIMD_NUM_INST];
    size_t op_count;
} SIMDDecoder;

/* 初始化 trie（构造函数调用一次） */
static inline __attribute__((unused)) void simd_decoder_init(SIMDDecoder *d)
{
    if (d->op_count > 0) return;
    int idx = 0;
#define X(name, pattern, op_type, imm_fn)                                                                 \
    d->op_table[idx].mask = pattern_to_mask(pattern), d->op_table[idx].match = pattern_to_match(pattern), \
    d->op_table[idx].exec = NULL, d->op_table[idx].type = op_type, idx++
    {
        INSTRUCTION_LIST;
    }
#undef X
    d->op_count = idx;
    decode_trie_build(&d->trie, d->op_table, d->op_count);
}

#endif /* SIMD_DISPATCH_MOD_H */
