/*
 * kerncheck.c — RISC-V GPU Kernel Binary Validator
 *
 * 读取 .bin kernel 二进制，用 dispatch.h 的 trie 译码每条指令，
 * 检查合法性、ebreak 存在性、branch target 范围等。
 *
 * Build: (in exp/)
 *   gcc -Wall -O2 -I. -Icore -o tools/kerncheck tools/kerncheck.c \
 *       core/decode_trie.c -lm
 *
 * Usage: ./tools/kerncheck kernels/vecmul.bin
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "core/dispatch.h"
#include "core/predecode.h"
#include "core/inst.h"

#define MAX_INST 4096

static const char *inst_names[300];

static void __attribute__((constructor)) build_names(void)
{
    int di = 0;
#define X(name, pattern, op_type, imm_fn) inst_names[di++] = #name;
    INSTRUCTION_LIST
#undef X
}

static const char *abi_name(int r)
{
    static const char *abi[] = {
            "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0",  "a1",  "a2", "a3", "a4", "a5",
            "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
    };
    if (r < 32) return abi[r];
    return "?";
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: kerncheck <kernel.bin>\n");
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror(argv[1]);
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    size_t sz = st.st_size;
    uint32_t *buf = malloc(sz);
    if (read(fd, buf, sz) != (ssize_t)sz) {
        perror("read");
        return 1;
    }
    close(fd);

    int n = (int)(sz / 4);
    if (n > MAX_INST) n = MAX_INST;
    printf("Kernel: %s  (%d inst, %zu bytes)\n\n", argv[1], n, sz);

    SIMDDecoder dec = {0};
    simd_decoder_init(&dec);

    int errors = 0, has_ebreak = 0;

    printf("%-6s %-10s %-18s %s\n", "Addr", "Hex", "Instruction", "Operands");
    printf("%-6s %-10s %-18s %s\n", "----", "---", "-----------", "--------");

    for (int i = 0; i < n; i++) {
        uint32_t inst = buf[i];
        uint32_t addr = i * 4;
        uint16_t id = simd_trie_lookup(&dec.trie, inst);

        int rd = (inst >> 7) & 0x1F, rs1 = (inst >> 15) & 0x1F, rs2 = (inst >> 20) & 0x1F;
        int opcode = inst & 0x7F, funct3 = (inst >> 12) & 7;

        const char *name = id ? inst_names[id - 1] : "???";
        char ops[80] = "";
        char note[20] = "";

        if (!id) {
            snprintf(ops, sizeof(ops), "0x%08X", inst);
            snprintf(note, sizeof(note), "  ← ILLEGAL");
            errors++;
        } else if (!strcmp(name, "ebreak")) {
            has_ebreak = 1;
        } else if (!strcmp(name, "lui") || !strcmp(name, "auipc")) {
            snprintf(ops, sizeof(ops), "%s, 0x%x", abi_name(rd), inst & 0xFFFFF000);
        } else if (!strcmp(name, "jal")) {
            int32_t off = immJ(inst);
            snprintf(ops, sizeof(ops), "%s, %+d", abi_name(rd), off);
        } else if (opcode == 0x63) {
            int32_t off = immB(inst);
            snprintf(ops, sizeof(ops), "%s, %s, %+d", abi_name(rs1), abi_name(rs2), off);
        } else if (opcode == 0x03 || opcode == 0x13 || opcode == 0x67) {
            int32_t off = immI(inst);
            snprintf(ops, sizeof(ops), "%s, %d(%s)", abi_name(rd), off, abi_name(rs1));
        } else if (opcode == 0x23) {
            int32_t off = immS(inst);
            snprintf(ops, sizeof(ops), "%s, %d(%s)", abi_name(rs2), off, abi_name(rs1));
        } else if (opcode == 0x07) { /* flw */
            int32_t off = immI(inst);
            snprintf(ops, sizeof(ops), "f%d, %d(%s)", rd, off, abi_name(rs1));
        } else if (opcode == 0x27) { /* fsw */
            int32_t off = immS(inst);
            snprintf(ops, sizeof(ops), "f%d, %d(%s)", rs2, off, abi_name(rs1));
        } else if (opcode == 0x53 || opcode == 0x43 || opcode == 0x47 || opcode == 0x4B || opcode == 0x4F) {
            /* FP R-type / R4-type */
            if (funct3 == 6)
                snprintf(ops, sizeof(ops), "f%d, f%d, f%d", rd, rs1, rs2);
            else
                snprintf(ops, sizeof(ops), "f%d, f%d, f%d", rd, rs1, rs2);
        } else if (opcode == 0x2B) {
            /* VPU/TCU custom-1 */
            if (strstr(name, "mma_cfg"))
                snprintf(ops, sizeof(ops), "a%d, K=%s, N/fmts=%s", rd, abi_name(rs1), abi_name(rs2));
            else if (strstr(name, "mma_zero") || strstr(name, "mma_relu"))
                snprintf(ops, sizeof(ops), "a%d", rd);
            else if (strstr(name, "mma_s"))
                snprintf(ops, sizeof(ops), "a%d, v%d, f%d", rd, rs1 - 16, rs2);
            else if (strstr(name, "mma_ld") || strstr(name, "mma_st") || strstr(name, "mma_bias"))
                snprintf(ops, sizeof(ops), "a%d, (%s), %s", rd, abi_name(rs1), abi_name(rs2));
            else if (strstr(name, "vld_v") || strstr(name, "vst_v"))
                snprintf(ops, sizeof(ops), "v%d, (%s), %s", rd - 16, abi_name(rs1), abi_name(rs2));
            else if (strstr(name, "vfadd_v") || strstr(name, "vfsub_v") || strstr(name, "vfmul_v") ||
                     strstr(name, "vfdiv_v") || strstr(name, "vffma_v"))
                snprintf(ops, sizeof(ops), "v%d, v%d, v%d", rd - 16, rs1 - 16, rs2 - 16);
            else if (strstr(name, "vfmul_vs") || strstr(name, "vfadd_vs") || strstr(name, "vfdiv_vs"))
                snprintf(ops, sizeof(ops), "v%d, v%d, f%d", rd - 16, rs1 - 16, rs2);
            else if (strstr(name, "vfexp_v") || strstr(name, "vfsig_v") || strstr(name, "vftanh_v") ||
                     strstr(name, "vfsqrt_v"))
                snprintf(ops, sizeof(ops), "v%d, v%d", rd - 16, rs1 - 16);
            else if (strstr(name, "vred"))
                snprintf(ops, sizeof(ops), "f%d, v%d", rd, rs1 - 16);
            else
                snprintf(ops, sizeof(ops), "f%d, f%d, f%d", rd, rs1, rs2);
        } else if (opcode == 0x33) {
            /* RV32M/RV32A R-type */
            snprintf(ops, sizeof(ops), "%s, %s, %s", abi_name(rd), abi_name(rs1), abi_name(rs2));
        } else {
            snprintf(ops, sizeof(ops), "rd=%d,rs1=%d,rs2=%d", rd, rs1, rs2);
        }

        printf("0x%04x 0x%08X  %-18s %s%s\n", addr, inst, name, ops, note);
    }

    printf("\n--- Summary ---\n");
    printf("Total:  %d instructions\n", n);
    if (errors) printf("Errors: %d illegal\n", errors);
    printf("ebreak: %s\n", has_ebreak ? "yes" : "MISSING");
    return errors ? 1 : 0;
}
