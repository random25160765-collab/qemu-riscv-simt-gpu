/*
 * fusion.c — DFG pattern-matching instruction fusion
 *
 * 从 STORE 指令反向追 use-def 链，匹配已知拓扑模式。
 * 命中 → 替换为 op_fused_*, 未命中 → 保持原样。
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inst.h"
#include "fusion.h"

/* ============================================================
 * 调试开关: 设为 1 开启融合过程追踪 (stdout)
 * ============================================================ */
#define FUSION_TRACE 1

#if FUSION_TRACE
#define TRACE(fmt, ...) fprintf(stdout, "[fusion] " fmt, ##__VA_ARGS__)
#else
#define TRACE(fmt, ...) ((void)0)
#endif

/* ============================================================
 * DAG 节点类型
 * ============================================================ */
typedef enum { N_OTHER, N_LOAD, N_STORE, N_COMPUTE, N_CONTROL } NodeKind;

typedef struct {
    int16_t rd;       /* dest register, -1 = none */
    int16_t rs1, rs2; /* source registers */
    NodeKind kind;
    bool visited;
    uint32_t inst; /* raw instruction for pattern matching */
} DFNode;

/* opcode helpers */
static NodeKind classify(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    if (op == 0x63) return N_CONTROL;               /* branches */
    if (op == 0x6F || op == 0x67) return N_CONTROL; /* jal/jalr */
    if (op == 0x73) return N_CONTROL;               /* csr/ebreak */
    if (op == 0x23) return N_STORE;                 /* sb/sh/sw */
    if (op == 0x27) return N_STORE;                 /* fsw */
    if (op == 0x03) return N_LOAD;                  /* lb/lh/lw/lbu/lhu */
    if (op == 0x07) return N_LOAD;                  /* flw */
    return N_COMPUTE;
}

static int reg_rd(uint32_t inst)  { return (int)((inst >> 7) & 0x1F); }
static int reg_rs1(uint32_t inst) { return (int)((inst >> 15) & 0x1F); }
static int reg_rs2(uint32_t inst) { return (int)((inst >> 20) & 0x1F); }

static bool has_rd(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    return (op != 0x23 && op != 0x27 && op != 0x63); /* not S/B type */
}
static bool has_rs1(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    return (op != 0x37 && op != 0x17 && op != 0x6F); /* not U/J */
}
static bool has_rs2(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    return (op == 0x33 || op == 0x23 || op == 0x27 || op == 0x63 || op == 0x53 ||
            op == 0x43 || op == 0x47 || op == 0x4B || op == 0x4F);
}

/* rd 是 FP 目的寄存器时需偏移到 last_writer[32..63] */
static bool is_fp_rd(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    return (op == 0x07) || (op == 0x53) || (op == 0x43) ||
           (op == 0x47) || (op == 0x4B) || (op == 0x4F);
}

/* rs1 是 FP 源寄存器时需偏移 (排除 fmv_w_x/ fcvt_s_w 等 GPR→FPR 指令) */
static bool is_fp_rs1(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    if (op != 0x53 && op != 0x43 && op != 0x47 && op != 0x4B && op != 0x4F) return false;
    /* fmv_w_x (f7=0x78) & fcvt_s_w / fcvt_s_wu (f7=0x68): rs1 是 GPR */
    uint32_t f7 = (inst >> 25) & 0x7F;
    if (f7 == 0x78 || f7 == 0x68) return false;
    return true;
}

/* rs2 是 FP 源寄存器时需偏移 (FP compute + FSW) */
static bool is_fp_rs2(uint32_t inst)
{
    uint32_t op = inst & 0x7F;
    if (op == 0x27) return true; /* FSW: rs2 = FPR value */
    if (op != 0x53 && op != 0x43 && op != 0x47 && op != 0x4B && op != 0x4F) return false;
    /* fmv_x_w (f7=0x70, rs2 unused) — has_rs2 returns true for 0x53 but rs2 is imm */
    return true;
}

/* ============================================================
 * DFG 构建: ThOp[] → DFNode[], 建立 def-use 关系
 * ============================================================ */
static void build_dag(ThOp *code, int n, DFNode *nodes)
{
    int16_t last_writer[64] = {0}; /* 0..31 GPR, 32..63 FPR */
    for (int i = 0; i < 64; i++)
        last_writer[i] = -1;

    for (int i = 0; i < n; i++) {
        uint32_t inst = code[i].inst;
        nodes[i].inst    = inst;
        nodes[i].kind    = classify(inst);
        nodes[i].visited = false;
        nodes[i].rd      = -1;
        nodes[i].rs1     = -1;
        nodes[i].rs2     = -1;

        /* ★ 先读 rs1/rs2 (读旧的 last_writer), 再写 rd (更新 last_writer) ★
         * 否则 FMUL 这种 rd==rs1 的指令会自引用 */
        if (has_rs1(inst)) {
            int r    = reg_rs1(inst);
            int ridx = is_fp_rs1(inst) ? (32 + r) : r;
            nodes[i].rs1 = last_writer[ridx];
        }

        if (has_rs2(inst)) {
            int r    = reg_rs2(inst);
            int ridx = is_fp_rs2(inst) ? (32 + r) : r;
            nodes[i].rs2 = last_writer[ridx];
        }

        /* rd: 记录本指令写入的寄存器 (必须在 rs1/rs2 之后!) */
        if (has_rd(inst)) {
            int rd   = reg_rd(inst);
            int ridx = is_fp_rd(inst) ? (32 + rd) : rd;
            nodes[i].rd = (int16_t)ridx;
            last_writer[ridx] = (int16_t)i;
        }
    }
}

/* ============================================================
 * 从 STORE 反向 DFS, 收集可融合链上的节点索引
 * ============================================================ */
static int trace_chain(DFNode *nodes, int n_nodes, int start, int *chain, int max_len)
{
    int len = 0;
#define DFS_STACK_MAX 128
    int stack[DFS_STACK_MAX];
    int sp = 0;
    bool *in_chain = calloc(n_nodes, sizeof(bool));
    if (!in_chain) return 0;

    stack[sp++] = start;
    while (sp > 0 && len < max_len) {
        int cur = stack[--sp];
        if (cur < 0 || nodes[cur].visited || in_chain[cur]) continue;
        if (nodes[cur].kind == N_CONTROL) continue;

        in_chain[cur] = true;
        chain[len++] = cur;
#if FUSION_TRACE
        {
            uint32_t op = nodes[cur].inst & 0x7F;
            TRACE("    chain[%d]=%d op=0x%02x kind=%d rs1=%d rs2=%d\n",
                  len-1, cur, op, nodes[cur].kind, nodes[cur].rs1, nodes[cur].rs2);
        }
#endif

        if (nodes[cur].rs1 >= 0 && !in_chain[nodes[cur].rs1]) {
            if (sp >= DFS_STACK_MAX) break;
            stack[sp++] = nodes[cur].rs1;
        }
        if (nodes[cur].rs2 >= 0 && !in_chain[nodes[cur].rs2]) {
            if (sp >= DFS_STACK_MAX) break;
            stack[sp++] = nodes[cur].rs2;
        }
    }
    free(in_chain);
    return len;
}

/* 检查链中是否包含 SLLI shamt=2 (数组索引寻址, 区分向量/标量) */
static bool has_array_indexing(ThOp *code, int *chain, int len)
{
    for (int i = 0; i < len; i++) {
        uint32_t inst = code[chain[i]].inst;
        /* SLLI: opcode=0010011, funct3=001, shamt in bits[25:20] */
        if ((inst & 0x7F) == 0x13 && ((inst >> 12) & 7) == 1 && ((inst >> 20) & 0x1F) == 2)
            return true;
    }
    return false;
}

/* ============================================================
 * 模式匹配: 已收集的链是否匹配已知融合模式
 *   返回 fused 指令的 INSTRUCTION_LIST ID (0 = 不匹配)
 *   同时填充 params[]
 * ============================================================ */
static int match_pattern(DFNode *nodes, ThOp *code, int *chain, int len, int32_t params[4])
{
    (void)nodes;
    int n_load = 0, n_store = 0, n_fmul = 0, n_fmadd = 0;
    for (int i = 0; i < len; i++) {
        uint32_t inst = code[chain[i]].inst;
        uint32_t op = inst & 0x7F;
        if (op == 0x07) n_load++;  /* flw */
        if (op == 0x27) n_store++; /* fsw */
        if (op == 0x53) {
            uint32_t f7 = (inst >> 25) & 0x7F;
            if (f7 == 0x08) n_fmul++; /* fmul.s: funct7=0001000 */
        }
        /* FMA: opcodes 0x43/0x47/0x4B/0x4F */
        if (op == 0x43 || op == 0x47 || op == 0x4B || op == 0x4F) n_fmadd++;
    }

    TRACE("  match: loads=%d fmul=%d fmadd=%d stores=%d has_slli=%d\n",
          n_load, n_fmul, n_fmadd, n_store, has_array_indexing(code, chain, len));

    /* Pattern: vecmul = 2 FLW + 1 FMUL + 1 FSW, with array indexing
     *   (also covers scal_mul variant with scalar at 0x400000) */
    if (n_load == 2 && n_fmul == 1 && n_store == 1 && has_array_indexing(code, chain, len)) {
        for (int i = 0; i < len; i++) {
            int idx = chain[i];
            uint32_t inst = code[idx].inst;
            if ((inst & 0x7F) == 0x37) { /* LUI */
                uint32_t val = inst & 0xFFFFF000;
                if (val == 0x10000000) params[0] = 0x100000;
                if (val == 0x20000000) params[1] = 0x200000;
                if (val == 0x30000000) params[2] = 0x300000;
            }
        }
        /* scalar at 0x400000 → scal_mul variant */
        for (int i = 0; i < len; i++) {
            int idx = chain[i];
            uint32_t inst = code[idx].inst;
            if ((inst & 0x7F) == 0x37 && (inst & 0xFFFFF000) == 0x40000000)
                return 3; /* FUSED_SCAL_MUL */
        }
        return 1; /* FUSED_VECMUL */
    }

    /* Pattern: gelu — uses fsigmoid + fmul (2×) + fsw, with array indexing */
    {
        int n_fsigmoid = 0;
        for (int i = 0; i < len; i++) {
            uint32_t inst = code[chain[i]].inst;
            if ((inst & 0x7F) == 0x53) {
                uint32_t f7 = (inst >> 25) & 0x7F;
                uint32_t rs2 = (inst >> 20) & 0x1F;
                if (f7 == 0x30 && rs2 == 5) n_fsigmoid++; /* fsigmoid.s: rs2=5 */
            }
        }
        if (n_fsigmoid >= 1 && n_fmul >= 2 && n_store == 1 && has_array_indexing(code, chain, len)) {
            for (int i = 0; i < len; i++) {
                int idx = chain[i];
                uint32_t inst = code[idx].inst;
                if ((inst & 0x7F) == 0x37) {
                    uint32_t val = inst & 0xFFFFF000;
                    if (val == 0x10000000) params[0] = 0x100000;
                    if (val == 0x20000000) params[1] = 0x200000;
                }
            }
            return 4; /* FUSED_GELU */
        }
    }

    /* Pattern: softmax body = FLW + FEXP + FADD + FDIV + FSW, with array indexing */
    {
        int n_fexp = 0, n_fadd = 0, n_fdiv = 0;
        for (int i = 0; i < len; i++) {
            uint32_t inst = code[chain[i]].inst;
            if ((inst & 0x7F) == 0x53) {
                uint32_t f7 = (inst >> 25) & 0x7F;
                uint32_t rs2 = (inst >> 20) & 0x1F;
                if (f7 == 0x30 && rs2 == 0) n_fexp++; /* fexp.s: rs2=0 */
                if (f7 == 0x00) n_fadd++;             /* fadd.s */
                if (f7 == 0x0C) n_fdiv++;             /* fdiv.s */
            }
        }
        if (n_fexp >= 2 && n_fadd >= 1 && n_fdiv >= 1 && n_store == 1 &&
            has_array_indexing(code, chain, len)) {
            for (int i = 0; i < len; i++) {
                int idx = chain[i];
                uint32_t inst = code[idx].inst;
                if ((inst & 0x7F) == 0x37) {
                    uint32_t val = inst & 0xFFFFF000;
                    if (val == 0x10000000) params[0] = 0x100000;
                    if (val == 0x20000000) params[1] = 0x200000;
                }
            }
            return 5; /* FUSED_SOFTMAX */
        }
    }

    /* Pattern: matmul body = 2 FLW + 1 FMADD (single inner-loop iteration) */
    if (n_load == 2 && n_fmadd == 1 && n_store == 0) {
        for (int i = 0; i < len; i++) {
            int idx = chain[i];
            uint32_t inst = code[idx].inst;
            if ((inst & 0x7F) == 0x37) { /* LUI */
                uint32_t val = inst & 0xFFFFF000;
                if (val == 0x10000000) params[0] = 0x100000;
                if (val == 0x20000000) params[1] = 0x200000;
            }
        }
        return 2; /* FUSED_LD2_FMA */
    }

    return 0;
}

/* ============================================================
 * fusion_pass: 主入口
 *   1. 建 DAG  2. 从 STORE 追链  3. 模式匹配  4. 生成 code_out
 * ============================================================ */
ThOp *fusion_pass(ThOp *code_in, int tcount_in, int *tcount_out)
{
    if (tcount_in < 1) {
        *tcount_out = 0;
        return NULL;
    }

    DFNode *nodes = calloc(tcount_in, sizeof(DFNode));
    build_dag(code_in, tcount_in, nodes);

    /* 找到可融合链 */
    int *chain = calloc(tcount_in, sizeof(int));
    int *map   = calloc(tcount_in, sizeof(int)); /* old_idx → new_idx (-1 = 融合掉) */
    for (int i = 0; i < tcount_in; i++)
        map[i] = i;

    int fused_count = 0;
    int n_stores    = 0;
    for (int i = 0; i < tcount_in; i++)
        if (nodes[i].kind == N_STORE) n_stores++;
    TRACE("fusion_pass: %d ops, %d stores\n", tcount_in, n_stores);

    for (int i = 0; i < tcount_in; i++) {
        if (nodes[i].kind != N_STORE || nodes[i].visited) continue;

        int c_len = trace_chain(nodes, tcount_in, i, chain, 64);
        TRACE("  store@%d (0x%08x) chain_len=%d\n", i, code_in[i].inst, c_len);
        if (c_len < 4) continue; /* 至少 LOAD+LOAD+COMPUTE+STORE */

        int32_t params[4] = {0};
        int fid = match_pattern(nodes, code_in, chain, c_len, params);
        if (fid == 0) continue;

        /* 标记链中节点为已访问 (不重复融合) */
        for (int j = 0; j < c_len; j++)
            nodes[chain[j]].visited = true;

        /* 找到链中第一个 FP load (FLW) 的索引作为融合起点,
         * 保留之前的 CTRL 读取和全局 ID 计算 */
        int c_min = tcount_in, c_max = 0;
        for (int j = 0; j < c_len; j++) {
            if (nodes[chain[j]].kind == N_LOAD && chain[j] < c_min)
                c_min = chain[j];
            if (chain[j] > c_max) c_max = chain[j];
        }
        if (c_min == tcount_in) {
            for (int j = 0; j < c_len; j++)
                if (chain[j] < c_min) c_min = chain[j];
        }
        int span = c_max - c_min + 1; /* 实际跨越的指令数 */

        TRACE("  → fused! fid=%d c_min=%d c_max=%d span=%d\n", fid, c_min, c_max, span);

        /* 提取线程 ID 寄存器: 从链中找 SLLI (slli rx, tid_reg, 2), rs1=tid_reg */
        int tid_reg = -1;
        for (int j = 0; j < c_len; j++) {
            uint32_t inst = code_in[chain[j]].inst;
            if ((inst & 0x7F) == 0x13 && ((inst >> 12) & 7) == 1 && ((inst >> 20) & 0x1F) == 2) {
                tid_reg = (int)((inst >> 15) & 0x1F); /* SLLI rs1 = thread-id register */
                break;
            }
        }

        /* 更新 c_min 节点的信息为 fused */
        code_in[c_min].handler = (void *)(uintptr_t)(107 + fid);
        code_in[c_min].skip    = (int16_t)span;
        code_in[c_min].pc_advance = (int16_t)(span * 4);
        if (tid_reg >= 0) code_in[c_min].rs1 = (int8_t)tid_reg;
        memcpy(code_in[c_min].params, params, sizeof(params));

        /* 链中除 c_min 外的节点标记为 -1 (被融合) */
        for (int j = 0; j < c_len; j++)
            if (chain[j] != c_min) map[chain[j]] = -1;
        /* 不在链中但在 c_min..c_max 之间的节点也要移除 */
        for (int j = c_min + 1; j < c_max; j++)
            if (map[j] >= 0) map[j] = -1;
        fused_count++;
    }

    TRACE("fusion_pass: %d patterns fused\n", fused_count);

    /* matmul loop: detect backward branch + loop body (flw+flw+fmadd+addi+bne) */
    for (int i = 0; i < tcount_in && i < 120; i++) {
        int bt = code_in[i].branch_tgt;
        if (bt < 0 || bt >= i) continue;                /* not a backward branch */
        if ((code_in[i].inst & 0x7F) != 0x63) continue; /* not a branch instruction */

        int n_flw = 0, n_fmadd = 0, n_addi = 0;
        for (int j = bt; j <= i && j < tcount_in; j++) {
            uint32_t inst = code_in[j].inst;
            if ((inst & 0x7F) == 0x07) n_flw++;
            if ((inst & 0x7F) == 0x43) n_fmadd++;
            if ((inst & 0x7F) == 0x13 && ((inst >> 12) & 7) == 0) n_addi++;
        }
        if (n_flw >= 2 && n_fmadd >= 1 && n_addi >= 1) {
            TRACE("  matmul loop: bt=%d i=%d flw=%d fmadd=%d addi=%d\n", bt, i, n_flw, n_fmadd, n_addi);
            int c_min = bt;
            for (int j = bt + 1; j <= i; j++) map[j] = -1;
            if (i != c_min) map[i] = -1;

            code_in[c_min].handler = (void *)(uintptr_t)113; /* FUSED_MATMUL_LOOP */
            code_in[c_min].skip    = (int16_t)(i - bt + 1);
            code_in[c_min].pc_advance = (int16_t)((i - bt + 1) * 4);
            code_in[c_min].params[0] = 128; /* K */
            code_in[c_min].params[1] = 128; /* N */
            code_in[c_min].params[2] = 0x100000;
            code_in[c_min].params[3] = 0x200000;
            code_in[c_min].rs1 = code_in[bt].rs1; /* row */
            code_in[c_min].rs2 = code_in[bt].rs2; /* col */
            fused_count++;
            break;
        }
    }

    /* 构建 code_out */
    int out_n = 0;
    for (int i = 0; i < tcount_in; i++)
        if (map[i] >= 0) out_n++;

    ThOp *code_out = calloc(out_n + 1, sizeof(ThOp));
    int *remap     = calloc(tcount_in, sizeof(int));
    int out_idx    = 0;
    for (int i = 0; i < tcount_in; i++) {
        if (map[i] >= 0) {
            remap[i] = out_idx;
            code_out[out_idx++] = code_in[i];
        } else {
            remap[i] = -1;
        }
    }

    /* 更新 branch_tgt */
    for (int i = 0; i < out_n; i++) {
        int bt = code_out[i].branch_tgt;
        if (bt >= 0 && bt < tcount_in && remap[bt] >= 0)
            code_out[i].branch_tgt = remap[bt];
        else if (bt >= 0)
            code_out[i].branch_tgt = -1;
    }

    free(remap);
    free(chain);
    free(nodes);

    *tcount_out = out_n;
    return code_out;
}
