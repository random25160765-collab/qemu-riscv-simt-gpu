/*
 * fuzzer.c — VPU Engine Fuzzer v2
 * 随机 RISC-V ALU/FP/BR 指令, 多 mask 交叉验证, x0 不变量。
 * v2: 支持分支 + per-lane 分歧, 测试 SIMT stack。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "state.h"
#include "config.h"
#include "gpgpu_core.h"
#include "engine.h"
#include "memory.h"
#include "predecode.h"
#include "dispatch.h"

static uint32_t xs_state = 42;
static uint32_t rng(void)
{
    uint32_t x = xs_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return xs_state = x;
}

#define KERN_ADDR 0x1000
#define KERN_MAX 512

/* ---- safe ops (排除访存/CSR/融合/custom, 包含分支) ---- */
static struct {
    const char *pat_str;
} safe[128];
static int n_safe;
#undef X
#define X(op, pat, _t, _imm)                             \
    do {                                                 \
        static const char *bad[] = {"jalr",              \
                                    "lb",                \
                                    "lh",                \
                                    "lw",                \
                                    "lbu",               \
                                    "lhu",               \
                                    "sb",                \
                                    "sh",                \
                                    "sw",                \
                                    "flw",               \
                                    "fsw",               \
                                    "csrrw",             \
                                    "csrrs",             \
                                    "csrrc",             \
                                    "csrrwi",            \
                                    "csrrsi",            \
                                    "csrrci",            \
                                    "ebreak",            \
                                    "lr_w",              \
                                    "sc_w",              \
                                    "amoswap_w",         \
                                    "amoadd_w",          \
                                    "amoxor_w",          \
                                    "amoand_w",          \
                                    "amoor_w",           \
                                    "amomin_w",          \
                                    "amomax_w",          \
                                    "amominu_w",         \
                                    "amomaxu_w",         \
                                    "fused_ld2_fma",     \
                                    "fused_scal_mul",    \
                                    "fused_gelu",        \
                                    "fused_softmax",     \
                                    "fused_matmul_loop", \
                                    "tex",               \
                                    "barrier",           \
                                    "fused_vecmul",      \
                                    NULL};               \
        int ok = 1;                                      \
        for (int i = 0; bad[i]; i++)                     \
            if (!strcmp(#op, bad[i])) {                  \
                ok = 0;                                  \
                break;                                   \
            }                                            \
        if (ok && n_safe < 128) {                        \
            safe[n_safe].pat_str = pat;                  \
            n_safe++;                                    \
        }                                                \
    } while (0)
static void __attribute__((constructor)) _init(void)
{
    INSTRUCTION_LIST;
}

/* ---- 指令生成 ---- */
static uint32_t gen(const char *p)
{
    uint32_t v = 0;
    int b = 31, nb = 0;
    for (; *p; p++)
        if (*p != ' ') {
            if (*p == '1')
                v |= 1u << b;
            else if (*p == '?')
                v |= (rng() & 1) << b;
            b--;
            nb++;
        }
    if (nb != 32) fprintf(stderr, "GEN: bitcount=%d pat=%s\n", nb, p);
    if ((v & 0x7F) == 0) fprintf(stderr, "GEN: op=0 pat=%s\n", p);
    return v;
}

/* B-type immediate encoder (offset in bytes, must be even) */
static uint32_t enc_b_imm(int32_t off)
{
    uint32_t v = 0;
    v |= ((uint32_t)off >> 12) & 1;           /* imm[12] at bit 31 */
    v |= (((uint32_t)off >> 5) & 0x3F) << 25; /* imm[10:5] at bits 30:25 */
    v |= (((uint32_t)off >> 1) & 0xF) << 8;   /* imm[4:1] at bits 11:8 */
    v |= (((uint32_t)off >> 11) & 1) << 7;    /* imm[11] at bit 7 */
    return v;
}

/* J-type immediate encoder */
static uint32_t enc_j_imm(int32_t off)
{
    uint32_t v = 0;
    v |= ((uint32_t)off >> 20) & 1;            /* imm[20] at bit 31 */
    v |= (((uint32_t)off >> 1) & 0x3FF) << 21; /* imm[10:1] at bits 30:21 */
    v |= (((uint32_t)off >> 11) & 1) << 20;    /* imm[11] at bit 20 */
    v |= (((uint32_t)off >> 12) & 0xFF) << 12; /* imm[19:12] at bits 19:12 */
    return v;
}

int fuzzer_run(GPGPUState *s, uint32_t seed, int rounds, int verbose)
{
    xs_state = (seed != 0xFFFFFFFF) ? seed : (uint32_t)time(NULL);
    int err = 0;
    if (!n_safe) {
        fprintf(stderr, "FUZZ: no ops\n");
        return -1;
    }
    fprintf(stderr, "FUZZ: %d ops seed=0x%x n=%d\n", n_safe, xs_state, rounds);

    SIMDDecoder dec = {0};
    simd_decoder_init(&dec);

    for (int r = 0; r < rounds; r++) {
        memset(s->vram_ptr + KERN_ADDR, 0, KERN_MAX);
        memset(s->vram_ptr + 0x2000, 0, 0x1000);
        s->kernel.kernel_addr = KERN_ADDR;
        s->kernel.grid_dim[0] = s->kernel.grid_dim[1] = s->kernel.grid_dim[2] = 1;
        s->kernel.block_dim[0] = s->kernel.block_dim[1] = s->kernel.block_dim[2] = 1;

        /* generate instruction sequence */
        int ni = (int)(rng() % 20) + 5;
        uint32_t raw[32];
        for (int i = 0; i < ni; i++) {
            int op_idx = rng() % n_safe;
            raw[i] = gen(safe[op_idx].pat_str);
        }
        raw[ni] = 0x00100073; /* ebreak */

        /* fixup branch/jal targets: only forward (avoid infinite loops) */
        for (int i = 0; i < ni; i++) {
            uint32_t op = raw[i] & 0x7F;
            if ((op == 0x63 || op == 0x6F) && i + 1 <= ni) { /* branch or JAL, ensure forward */
                int tgt = (int)(rng() % (ni - i)) + i + 1;   /* i+1 .. ni */
                int32_t off = (tgt - i) * 4;
                if (op == 0x63)
                    raw[i] = (raw[i] & ~0xFE000F80) | enc_b_imm(off);
                else
                    raw[i] = (raw[i] & ~0xFFFFF000) | enc_j_imm(off);
            }
        }

        if (verbose > 1) {
            fprintf(stderr, "R%d insts(%d):", r, ni);
            for (int i = 0; i <= ni; i++)
                fprintf(stderr, " %08x", raw[i]);
            fprintf(stderr, "\n");
        }

        memcpy(s->vram_ptr + KERN_ADDR, raw, (ni + 1) * 4);

        int tc;
        ThOp *code = simd_predecode(&dec, s, KERN_ADDR, KERN_MAX, &tc);
        if (!code) {
            err++;
            continue;
        }

        /* SoA state: per-lane DIFFERENT initial values (制造分歧) */
        uint32_t ga[32 * 32], fa[32 * 32], pa[32], ma[32], ca[32];
        uint32_t gb[32 * 32], fb[32 * 32], pb[32], mb[32], cb[32];
        for (int l = 0; l < 32; l++) {
            pa[l] = pb[l] = KERN_ADDR;
            ma[l] = mb[l] = l;
            ca[l] = cb[l] = 0;
            for (int rg = 0; rg < 32; rg++) {
                /* seed each lane differently: combine rg, lane, and rng */
                uint32_t gv = rg ? (rng() ^ (l * 2654435761u)) : 0;
                uint32_t fv = rng() ^ (l * 0x9E3779B9u);
                ga[rg * 32 + l] = gb[rg * 32 + l] = gv;
                fa[rg * 32 + l] = fb[rg * 32 + l] = fv;
            }
        }

        EngineContext ca_ = {.s = s, .active = 0xFFFFFFFF, .thread_id = {0, 0, 0}, .block_id = {0, 0, 0}};
        EngineContext cb_ = {.s = s, .active = 0x1, .thread_id = {0, 0, 0}, .block_id = {0, 0, 0}};
        SIMTFrame sa[32], sb[32];
        int da = 0, db = 0;

        int ra = engine_exec(code, tc, &ca_, ga, fa, pa, ma, ca, sa, &da, -1);
        int rb = engine_exec(code, tc, &cb_, gb, fb, pb, mb, cb, sb, &db, -1);
        if (ra != 0 || rb != 0) {
            err++;
            goto nxt;
        }

        /* x0 invariant */
        for (int l = 0; l < 32; l++)
            if (ga[l] || gb[l]) {
                err++;
                goto nxt;
            }
        /* lane 0 cross-check */
        for (int rg = 1; rg < 32; rg++)
            if (ga[rg * 32] != gb[rg * 32] || fa[rg * 32] != fb[rg * 32]) {
                err++;
                goto nxt;
            }
    nxt:
        free(code);
        if (verbose && r % (rounds / 10 + 1) == 0) fprintf(stderr, " %d/%d e=%d\n", r + 1, rounds, err);
    }
    if (!err)
        fprintf(stderr, "FUZZ PASS %d\n", rounds);
    else
        fprintf(stderr, "FUZZ %d FAIL / %d\n", err, rounds);
    return err;
}

/* ---- replay mode: 直接执行指定 hex 指令序列 (供 bisect 使用) ---- */
int fuzzer_replay(GPGPUState *s, const char *hex_list, int verbose)
{
    /* 解析 hex 列表 */
    uint32_t raw[64];
    int ni = 0;
    const char *p = hex_list;
    while (*p && ni < 63) {
        while (*p == ' ')
            p++;
        if (!*p) break;
        raw[ni++] = (uint32_t)strtoul(p, (char **)&p, 16);
    }
    if (ni == 0) return -1;

    memset(s->vram_ptr + KERN_ADDR, 0, KERN_MAX);
    memcpy(s->vram_ptr + KERN_ADDR, raw, ni * 4);
    s->kernel.kernel_addr = KERN_ADDR;

    SIMDDecoder dec = {0};
    simd_decoder_init(&dec);
    int tc;
    ThOp *code = simd_predecode(&dec, s, KERN_ADDR, KERN_MAX, &tc);
    if (!code) return -1;

    uint32_t ga[32 * 32] = {0}, fa[32 * 32] = {0}, pa[32], ma[32], ca[32] = {0};
    uint32_t gb[32 * 32] = {0}, fb[32 * 32] = {0}, pb[32], mb[32], cb[32] = {0};
    for (int l = 0; l < 32; l++) {
        pa[l] = pb[l] = KERN_ADDR;
        ma[l] = mb[l] = l;
        for (int rg = 0; rg < 32; rg++) {
            uint32_t gv = rg ? (uint32_t)(l * 2654435761u) : 0;
            uint32_t fv = (uint32_t)(l * 0x9E3779B9u);
            ga[rg * 32 + l] = gb[rg * 32 + l] = gv;
            fa[rg * 32 + l] = fb[rg * 32 + l] = fv;
        }
    }

    EngineContext ca_ = {.s = s, .active = 0xFFFFFFFF, .thread_id = {0, 0, 0}, .block_id = {0, 0, 0}};
    EngineContext cb_ = {.s = s, .active = 0x1, .thread_id = {0, 0, 0}, .block_id = {0, 0, 0}};
    SIMTFrame sa[32], sb[32];
    int da = 0, db = 0;

    if (verbose) {
        fprintf(stderr, "REPLAY %d insts:", ni);
        for (int i = 0; i < ni; i++)
            fprintf(stderr, " %08x", raw[i]);
        fprintf(stderr, "\n");
    }

    /* trace mode: test each prefix to find crash point */
    if (verbose > 1) {
        for (int end = 0; end < ni; end++) {
            uint32_t sub_raw[64];
            int sn = end + 1;
            memcpy(sub_raw, raw, sn * 4);
            sub_raw[sn - 1] = 0x00100073; /* replace last with ebreak */
            memcpy(s->vram_ptr + KERN_ADDR, sub_raw, sn * 4);
            ThOp *sub = simd_predecode(&dec, s, KERN_ADDR, KERN_MAX, &tc);

            uint32_t sg[32 * 32], sf[32 * 32], sp[32], sm[32], sc[32];
            memcpy(sg, ga, sizeof(sg));
            memcpy(sf, fa, sizeof(sf));
            memcpy(sp, pa, sizeof(sp));
            memcpy(sm, ma, sizeof(sm));
            memset(sc, 0, sizeof(sc));
            SIMTFrame sstk[32];
            int ssd = 0;

            int r = engine_exec(sub, tc, &ca_, sg, sf, sp, sm, sc, sstk, &ssd, -1);
            fprintf(stderr, "  [%d] 0x%08x %s r=%d", end, raw[end], r == 0 ? "OK" : (r == -1 ? "ILLEGAL" : "CRASH?"),
                    r);
            if (r == 0) {
                fprintf(stderr, "  gpr[10]=0x%x fpr[10]=%f\n", sg[10 * 32 + 0], ((float *)sf)[10 * 32 + 0]);
            } else {
                fprintf(stderr, "\n");
            }
            free(sub);
        }
        free(code);
        return 0;
    }

    int ra = engine_exec(code, tc, &ca_, ga, fa, pa, ma, ca, sa, &da, -1);
    int rb = engine_exec(code, tc, &cb_, gb, fb, pb, mb, cb, sb, &db, -1);
    free(code);

    /* x0 check */
    for (int l = 0; l < 32; l++)
        if (ga[l] || gb[l]) return 1;
    /* lane 0 check */
    for (int rg = 1; rg < 32; rg++)
        if (ga[rg * 32] != gb[rg * 32] || fa[rg * 32] != fb[rg * 32]) return 1;

    return (ra != 0 || rb != 0) ? 1 : 0;
}
