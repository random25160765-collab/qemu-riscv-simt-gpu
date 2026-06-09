/*
 * decode.h — trie 基础设施 (SIMDDecoder + simd_decoder_init)
 *
 * 从旧 dispatch.h 拆分: 只含 trie 构造逻辑, INSTRUCTION_LIST 由外部提供。
 * engine.c 和 tools/kerncheck.c 各自 include 所需的 INSTRUCTION_LIST 来源。
 */
#ifndef SIMD_DECODE_H
#define SIMD_DECODE_H

#include "inst.h"
#include "utils.h"
#include "decode_trie.h"
#include "state.h"

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

static inline int32_t imm0(uint32_t i)
{
    (void)i;
    return 0;
}

#define SIMD_NUM_INST NUM_OF_INST

typedef struct {
    void *dispatch[SIMD_NUM_INST];
    int dispatch_ready;
    DecodeTrie trie;
    opcode_entry_t op_table[SIMD_NUM_INST];
    size_t op_count;
} SIMDDecoder;

static inline __attribute__((unused)) void simd_decoder_init(SIMDDecoder *d)
{
    if (d->op_count > 0) return;
    int idx = 0;
#define X(name, pattern, op_type, imm_fn)                                 \
    d->op_table[idx].mask = pattern_to_mask(pattern),                     \
    d->op_table[idx].match = pattern_to_match(pattern),                   \
    d->op_table[idx].exec = NULL, d->op_table[idx].type = op_type, idx++
    {
        INSTRUCTION_LIST;
    }
#undef X
    d->op_count = idx;
    decode_trie_build(&d->trie, d->op_table, d->op_count);
}

#endif /* SIMD_DECODE_H */
