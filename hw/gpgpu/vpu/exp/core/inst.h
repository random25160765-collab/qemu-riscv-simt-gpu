#ifndef INST_H
#define INST_H

#include "utils.h"

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
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
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
        } else if (*p == ' ') {
            p++;
            continue;
        }
        if (*p != ' ') bit--;
        p++;
    }

    return match;
}

/* ======== IMM ======== */
#define immI(i)   (SEXT(BITS(i, 31, 20), 12))
#define immU(i)   ((SEXT(BITS(i, 31, 12), 20) << 12))
#define immS(i)   ((SEXT(BITS(i, 31, 25), 7) << 5) | BITS(i, 11, 7))
#define immB(i)   ((SEXT(BITS(i, 31, 31), 1) << 12) | (BITS(i, 7, 7) << 11) | (BITS(i, 30, 25) << 5) | (BITS(i, 11, 8) << 1))
#define immJ(i)   ((SEXT(BITS(i, 31, 31), 1) << 20) | (BITS(i, 19, 12) << 12) | (BITS(i, 20, 20) << 11) | (BITS(i, 30, 21) << 1))
#define immCSR(i) (BITS(inst, 31, 20))

#endif /* INST_H */
