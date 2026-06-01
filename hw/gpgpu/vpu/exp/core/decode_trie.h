/*
 * decode_trie.h — 压缩前缀树指令解码器
 *
 * 在 32-bit 指令空间上构建 bit-by-bit 决策树，压缩单分支路径。
 * 配合 computed goto 实现 zero-overhead 线程化解释器。
 */

#ifndef DECODE_TRIE_H
#define DECODE_TRIE_H

#include <stdint.h>
#include <stdbool.h>

/* opcode_entry — 指令描述（与 gpgpu_core.c 中的定义保持一致） */
typedef void (*exec_func_t)(void *ctx, int lane_id);
typedef struct opcode_entry {
    uint32_t mask;
    uint32_t match;
    exec_func_t exec;
    int type;
} opcode_entry_t;

/*
 * ============================================================================
 * 压缩前缀树节点
 *
 * 两种形式：
 *   分支节点 — 检查 bit_pos 位，0→left, 1→right
 *   叶子节点 — 指向指令实现
 *
 * 子节点用 flat array 存储，idx 用 key 直接索引。
 * ============================================================================
 */
typedef struct TrieNode {
    /* 分支：检查的位位置 (0=LSB, 31=MSB, -1=叶子) */
    int8_t bit_pos;

    /* 分支：左右子节点索引 (0 = 非法指令) */
    uint16_t child[2];

    /* 叶子：computed goto label 地址，或指令索引 */
    uint16_t instr_id; /* 0..NUM_INST, 0 = illegal */
} TrieNode;

/*
 * ============================================================================
 * 解码器状态
 * ============================================================================
 */
typedef struct {
    TrieNode *nodes;     /* 节点数组 (连续内存) */
    uint16_t root_idx;   /* 根节点索引 */
    uint16_t num_nodes;  /* 实际节点数 */
    uint16_t max_nodes;  /* 预分配节点数 */
    uint16_t num_instrs; /* 指令总数 */
} DecodeTrie;

/*
 * ============================================================================
 * API
 * ============================================================================
 */

/* 从 opcode_entry 表构建压缩前缀树。
 * 返回 0 成功，-1 失败。 */
int decode_trie_build(DecodeTrie *trie, const opcode_entry_t *entries, size_t count);

/* 释放 */
void decode_trie_destroy(DecodeTrie *trie);

/* 快速查找：返回 instr_id (0=illegal, 1..N=合法指令) */
static inline uint16_t decode_trie_lookup(const DecodeTrie *trie, uint32_t inst)
{
    uint16_t idx = trie->root_idx;
    const TrieNode *nodes = trie->nodes;

    while (nodes[idx].bit_pos >= 0) {
        int bit = (inst >> nodes[idx].bit_pos) & 1;
        uint16_t next = nodes[idx].child[bit];
        if (next == 0) return 0; /* illegal */
        idx = next;
    }
    return nodes[idx].instr_id;
}

/* 打印统计 */
void decode_trie_stats(const DecodeTrie *trie);

#endif /* DECODE_TRIE_H */
