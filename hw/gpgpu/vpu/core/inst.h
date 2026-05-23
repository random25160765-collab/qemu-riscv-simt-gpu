#ifndef INST_H
#define INST_H

#include <stdint.h>
#include "platform/gpgpu_core.h"

#ifdef DEBUG_OPCODE_TABLE
#define IF_DEBUG_OPCODE_TABLE(code) code
#else
#define IF_DEBUG_OPCODE_TABLE(code) ((void)0)
#endif

#ifdef DEBUG_INST
#define IF_DEBUG_INST(code) code
#else
#define IF_DEBUG_INST(code) ((void)0)
#endif

#define NUM_OF_INST 300
#define MATCH_EBREAK 0x00100073

/* Instruction Parsing Macros and tools */
static inline uint32_t pattern_to_mask(const char *pattern) {
    uint32_t mask = 0;
    const char *p = pattern;
    int bit = 31;

    while (*p && bit >= 0) {
        if (*p == '0' || *p == '1') {
            mask |= (1U << bit);
        }
        p++;
        bit--;
    }
    return mask;
}

static inline uint32_t pattern_to_match(const char *pattern) {
    uint32_t match = 0;
    const char *p = pattern;
    int bit = 31;

    while (*p && bit >= 0) {
        if (*p == '1') {
            match |= (1U << bit);
        }
        p++;
        bit--;
    }
    return match;
}

/* Immediate extraction functions */
static inline int32_t immI(uint32_t inst) {
    return (int32_t)inst >> 20;
}

static inline int32_t immS(uint32_t inst) {
    int32_t imm = ((inst >> 25) << 5) | ((inst >> 7) & 0x1F);
    return (imm << 20) >> 20;
}

static inline int32_t immB(uint32_t inst) {
    int32_t imm = ((inst >> 31) << 12)
                | (((inst >> 7) & 1) << 11)
                | (((inst >> 25) & 0x3F) << 5)
                | (((inst >> 8) & 0xF) << 1);
    return (imm << 19) >> 19;
}

static inline int32_t immU(uint32_t inst) {
    return (int32_t)(inst & 0xFFFFF000);
}

static inline int32_t immJ(uint32_t inst) {
    int32_t imm = ((inst >> 31) << 20)
                | (((inst >> 12) & 0xFF) << 12)
                | (((inst >> 20) & 1) << 11)
                | (((inst >> 21) & 0x3FF) << 1);
    return (imm << 11) >> 11;
}

static inline int32_t immCSR(uint32_t inst) {
    return (inst >> 20) & 0xFFF;
}

#endif /* INST_H */