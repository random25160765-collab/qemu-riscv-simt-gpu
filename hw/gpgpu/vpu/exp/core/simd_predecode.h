/*
 * simd_predecode.h — ThOp 预译码
 */
#ifndef SIMD_PREDECODE_H
#define SIMD_PREDECODE_H

#include <stdint.h>
#include <stdlib.h>
#include "state.h"
#include "simd_dispatch.h"

/* ============================================================
 * ThOp — 预译码指令
 * ============================================================ */
typedef struct {
    void    *handler;
    uint32_t inst;
    int32_t  imm;
    int8_t   rd, rs1, rs2, rs3;
    int32_t  branch_tgt;
} ThOp;

/* trie 查找 */
static inline uint16_t simd_trie_lookup(const DecodeTrie *trie, uint32_t inst) {
    uint16_t idx = trie->root_idx;
    const TrieNode *nodes = trie->nodes;
    while (nodes[idx].bit_pos >= 0) {
        int bit = (inst >> nodes[idx].bit_pos) & 1;
        uint16_t next = nodes[idx].child[bit];
        if (next == 0) return 0;
        idx = next;
    }
    return nodes[idx].instr_id;
}

/* predecode: kernel .bin → ThOp 数组 */
static inline ThOp *simd_predecode(SIMDDecoder *d, GPGPUState *s,
                                    uint32_t kern_addr, size_t size, int *count) {
    int n = (int)(size / 4);
    ThOp *code = malloc((n + 1) * sizeof(ThOp));
    for (int i = 0; i < n; i++) {
        uint32_t pc   = kern_addr + i * 4;
        uint32_t inst = *(uint32_t *)(s->vram_ptr + pc);
        uint16_t id   = simd_trie_lookup(&d->trie, inst);
        code[i].inst   = inst;
        code[i].rd     = (int8_t)BITS(inst, 11, 7);
        code[i].rs1    = (int8_t)BITS(inst, 19, 15);
        code[i].rs2    = (int8_t)BITS(inst, 24, 20);
        code[i].rs3    = (int8_t)BITS(inst, 31, 27);
        code[i].branch_tgt = -1;
        if (id == 0) { code[i].handler = NULL; continue; }
        code[i].handler = (void*)(uintptr_t)id;  /* store instr_id, exec_warp maps to handler */
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
    *count = n;
    return code;
}

/* inline helper: resolve NULL handlers in ThOp array */
#define SIMD_RESOLVE_HANDLERS(code, tcount) do { \
    for (int _i = 0; _i <= (tcount); _i++) \
        if ((code)[_i].handler == NULL) \
            (code)[_i].handler = (_i < (tcount)) ? &&op_illegal : &&op_done; \
} while(0)

#endif /* SIMD_PREDECODE_H */
