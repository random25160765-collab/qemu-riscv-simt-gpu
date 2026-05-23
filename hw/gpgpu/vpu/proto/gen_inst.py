#!/usr/bin/env python3
"""
指令编码生成器
用法：python3 gen.py <输出路径>
示例：python3 gen.py ../include/pt_inst.h
"""

import sys

INSTRUCTIONS = [
    # === RV32I 控制/分支指令 ===
    # (编号, 名字, 参数个数, 分支跳转?)
    # jal:   [JAL]   rd=%d, imm=0x%x, pc=0x%x -> 0x%x         → 源操作数: rd, imm, pc
    (1,   "jal",      3, True),
    # jalr:  [JALR]  rd=%d, src1=0x%x, imm=0x%x, pc=0x%x -> .. → 源操作数: rd, src1, imm, pc
    (2,   "jalr",     4, True),
    # beq:   [BEQ]   src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (3,   "beq",      4, True),
    # bne:   [BNE]   src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (4,   "bne",      4, True),
    # blt:   [BLT]   src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (5,   "blt",      4, True),
    # bge:   [BGE]   src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (6,   "bge",      4, True),
    # bltu:  [BLTU]  src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (7,   "bltu",     4, True),
    # bgeu:  [BGEU]  src1=0x%x, src2=0x%x, imm=0x%x, pc=0x%x, taken=%d
    (8,   "bgeu",     4, True),

    # === RV32I 内存访问 ===
    # lb:    [LB]    rd=%d, addr=0x%x, val=0x%02x, sign-extended=.. → 源: rd, addr
    (9,   "lb",       2, False),
    # lh:    [LH]    rd=%d, addr=0x%x, val=0x%04x, sign-extended=.. → 源: rd, addr
    (10,  "lh",       2, False),
    # lw:    [LW]    rd=%d, addr=0x%x, val=0x%08x                   → 源: rd, addr
    (11,  "lw",       2, False),
    # lbu:   [LBU]   rd=%d, addr=0x%x, val=0x%02x                   → 源: rd, addr
    (12,  "lbu",      2, False),
    # lhu:   [LHU]   rd=%d, addr=0x%x, val=0x%04x                   → 源: rd, addr
    (13,  "lhu",      2, False),
    # sb:    [SB]    addr=0x%x, val=0x%02x                          → 源: addr, val
    (14,  "sb",       2, False),
    # sh:    [SH]    addr=0x%x, val=0x%04x                          → 源: addr, val
    (15,  "sh",       2, False),
    # sw:    [SW]    addr=0x%x, val=0x%08x                          → 源: addr, val
    (16,  "sw",       2, False),

    # === RV32I U-type ===
    # lui:   [LUI]   rd=%d, imm=0x%x, result=0x%x                  → 源: rd, imm
    (17,  "lui",      2, False),
    # auipc: [AUIPC] rd=%d, imm=0x%x, pc=0x%x, result=..           → 源: rd, imm, pc
    (18,  "auipc",    3, False),

    # === RV32I I-type 整数运算 ===
    # addi:  [ADDI]  rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (19,  "addi",     3, False),
    # slti:  [SLTI]  rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (20,  "slti",     3, False),
    # sltiu: [SLTIU] rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (21,  "sltiu",    3, False),
    # xori:  [XORI]  rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (22,  "xori",     3, False),
    # ori:   [ORI]   rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (23,  "ori",      3, False),
    # andi:  [ANDI]  rd=%d, src1=0x%x, imm=0x%x, result=..         → 源: rd, src1, imm
    (24,  "andi",     3, False),
    # slli:  [SLLI]  rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (25,  "slli",     3, False),
    # srli:  [SRLI]  rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (26,  "srli",     3, False),
    # srai:  [SRAI]  rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (27,  "srai",     3, False),

    # === RV32I R-type 整数运算 ===
    # add:   [ADD]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (28,  "add",      3, False),
    # sub:   [SUB]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (29,  "sub",      3, False),
    # sll:   [SLL]   rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (30,  "sll",      3, False),
    # slt:   [SLT]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (31,  "slt",      3, False),
    # sltu:  [SLTU]  rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (32,  "sltu",     3, False),
    # xor:   [XOR]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (33,  "xor",      3, False),
    # srl:   [SRL]   rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (34,  "srl",      3, False),
    # sra:   [SRA]   rd=%d, src1=0x%x, shamt=%d, result=..         → 源: rd, src1, shamt
    (35,  "sra",      3, False),
    # or:    [OR]    rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (36,  "or",       3, False),
    # and:   [AND]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (37,  "and",      3, False),

    # === RV32M 乘除 ===
    # mul:   [MUL]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (38,  "mul",      3, False),
    # mulh:  [MULH]  rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (39,  "mulh",     3, False),
    # mulhsu:[MULHSU]rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (40,  "mulhsu",   3, False),
    # mulhu: [MULHU] rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (41,  "mulhu",    3, False),
    # div:   [DIV]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (42,  "div",      3, False),
    # divu:  [DIVU]  rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (43,  "divu",     3, False),
    # rem:   [REM]   rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (44,  "rem",      3, False),
    # remu:  [REMU]  rd=%d, src1=0x%x, src2=0x%x, result=..        → 源: rd, src1, src2
    (45,  "remu",     3, False),

    # === CSR 指令 ===
    # csrrw: [CSRRW] rd=%d, csr=0x%x, src1=0x%x, old_val=.., new_val=.. → 源: rd, csr, src1
    (46,  "csrrw",    3, False),
    # csrrs: [CSRRS] rd=%d, csr=0x%x, src1=0x%x, old_val=.., new_val=.. → 源: rd, csr, src1
    (47,  "csrrs",    3, False),
    # csrrc: [CSRRC] rd=%d, csr=0x%x, src1=0x%x, old_val=.., new_val=.. → 源: rd, csr, src1
    (48,  "csrrc",    3, False),
    # csrrwi:[CSRRWI]rd=%d, csr=0x%x, uimm=%d                          → 源: rd, csr, uimm
    (49,  "csrrwi",   3, False),
    # csrrsi:[CSRRSI]rd=%d, csr=0x%x, uimm=%d                          → 源: rd, csr, uimm
    (50,  "csrrsi",   3, False),
    # csrrci:[CSRRCI]rd=%d, csr=0x%x, imm=%d, old_val=.., new_val=..   → 源: rd, csr, imm
    (51,  "csrrci",   3, False),

    # === 系统指令 ===
    # ecall: (预留，暂无实现)
    (52,  "ecall",    0, False),
    # ebreak:[EBREAK] breakpoint instruction executed → 无参数
    (53,  "ebreak",   0, False),

    # === RV32F 访存 ===
    # flw:   [FLW]   rd=%d, addr=0x%x                                → 源: rd, addr
    (54,  "flw",      2, False),
    # fsw:   [FSW]   addr=0x%x, val=0x%08x                           → 源: addr, val
    (55,  "fsw",      2, False),

    # === RV32F 乘加 (F4 type, 4 registers) ===
    # fmadd_s: [FMADD_S]  rd=%d, rs1=.., rs2=.., rs3=..              → 源: rd, rs1, rs2, rs3
    (56,  "fmadd_s",  4, False),
    # fmsub_s: [FMSUB_S]  rd=%d, rs1=.., rs2=.., rs3=..              → 源: rd, rs1, rs2, rs3
    (57,  "fmsub_s",  4, False),
    # fnmsub_s:[FNMSUB_S] rd=%d, rs1=.., rs2=.., rs3=..              → 源: rd, rs1, rs2, rs3
    (58,  "fnmsub_s", 4, False),
    # fnmadd_s:[FNMADD_S] rd=%d, rs1=.., rs2=.., rs3=..              → 源: rd, rs1, rs2, rs3
    (59,  "fnmadd_s", 4, False),

    # === RV32F 算术 ===
    # fadd_s: [FADD_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (60,  "fadd_s",   3, False),
    # fsub_s: [FSUB_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (61,  "fsub_s",   3, False),
    # fmul_s: [FMUL_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (62,  "fmul_s",   3, False),
    # fdiv_s: [FDIV_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (63,  "fdiv_s",   3, False),
    # fsqrt_s:[FSQRT_S] rd=%d, rs1=..                                → 源: rd, rs1
    (64,  "fsqrt_s",  2, False),

    # === RV32F 符号注入 ===
    # fsgnj_s: [FSGNJ_S]  rd=%d, rs1=.., rs2=..                      → 源: rd, rs1, rs2
    (65,  "fsgnj_s",  3, False),
    # fsgnjn_s:[FSGNJN_S] rd=%d, rs1=.., rs2=..                      → 源: rd, rs1, rs2
    (66,  "fsgnjn_s", 3, False),
    # fsgnjx_s:[FSGNJX_S] rd=%d, rs1=.., rs2=..                      → 源: rd, rs1, rs2
    (67,  "fsgnjx_s", 3, False),

    # === RV32F 最值 ===
    # fmin_s: [FMIN_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (68,  "fmin_s",   3, False),
    # fmax_s: [FMAX_S]  rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (69,  "fmax_s",   3, False),

    # === RV32F 转换 (float <-> int) ===
    # fcvt_w_s: [FCVT_W_S]  rd=%d, rs1=..                            → 源: rd, rs1
    (70,  "fcvt_w_s",  2, False),
    # fcvt_wu_s:[FCVT_WU_S] rd=%d, rs1=..                            → 源: rd, rs1
    (71,  "fcvt_wu_s", 2, False),
    # fmv_x_w: [FMV_X_W]  rd=%d, rs1=..                              → 源: rd, rs1
    (72,  "fmv_x_w",  2, False),
    # feq_s:  [FEQ_S]   rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (73,  "feq_s",    3, False),
    # flt_s:  [FLT_S]   rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (74,  "flt_s",    3, False),
    # fle_s:  [FLE_S]   rd=%d, rs1=.., rs2=..                        → 源: rd, rs1, rs2
    (75,  "fle_s",    3, False),
    # fclass_s:[FCLASS_S] rd=%d, rs1=..                              → 源: rd, rs1
    (76,  "fclass_s", 2, False),
    # fcvt_s_w: [FCVT_S_W]  rd=%d, rs1=..                            → 源: rd, rs1
    (77,  "fcvt_s_w",  2, False),
    # fcvt_s_wu:[FCVT_S_WU] rd=%d, rs1=..                            → 源: rd, rs1
    (78,  "fcvt_s_wu", 2, False),
    # fmv_w_x: [FMV_W_X]  rd=%d, rs1=..                              → 源: rd, rs1
    (79,  "fmv_w_x",  2, False),

    # === LP Float 转换 ===
    # fcvt_s_bf16: [FCVT_S_BF16] rd=%d, rs1=..                       → 源: rd, rs1
    (80,  "fcvt_s_bf16",  2, False),
    # fcvt_bf16_s: [FCVT_BF16_S] rd=%d, rs1=..                       → 源: rd, rs1
    (81,  "fcvt_bf16_s",  2, False),
    # fcvt_s_e4m3: [FCVT_S_E4M3] rd=%d, rs1=..                       → 源: rd, rs1
    (82,  "fcvt_s_e4m3",  2, False),
    # fcvt_e4m3_s: [FCVT_E4M3_S] rd=%d, rs1=..                       → 源: rd, rs1
    (83,  "fcvt_e4m3_s",  2, False),
    # fcvt_s_e5m2: [FCVT_S_E5M2] rd=%d, rs1=..                       → 源: rd, rs1
    (84,  "fcvt_s_e5m2",  2, False),
    # fcvt_e5m2_s: [FCVT_E5M2_S] rd=%d, rs1=..                       → 源: rd, rs1
    (85,  "fcvt_e5m2_s",  2, False),
    # fcvt_s_e2m1: [FCVT_S_E2M1] rd=%d, rs1=..                       → 源: rd, rs1
    (86,  "fcvt_s_e2m1",  2, False),
    # fcvt_e2m1_s: [FCVT_E2M1_S] rd=%d, rs1=..                       → 源: rd, rs1
    (87,  "fcvt_e2m1_s",  2, False),

    # === 自定义科学函数 (funct7=0110000) ===
    # fexp.s:      e^x                                                → 源: rd, rs1
    (88,  "fexp_s",      2, False),
    # fln.s:       ln(x)                                              → 源: rd, rs1
    (89,  "fln_s",       2, False),
    # frcp.s:      1/x                                                → 源: rd, rs1
    (90,  "frcp_s",      2, False),
    # frsqrt.s:    1/sqrt(x)                                          → 源: rd, rs1
    (91,  "frsqrt_s",    2, False),
    # ftanh.s:     tanh(x)                                            → 源: rd, rs1
    (92,  "ftanh_s",     2, False),
    # fsigmoid.s:  1/(1+e^{-x})                                       → 源: rd, rs1
    (93,  "fsigmoid_s",  2, False),
    # fsin.s:      sin(x)                                             → 源: rd, rs1
    (94,  "fsin_s",      2, False),
    # fcos.s:      cos(x)                                             → 源: rd, rs1
    (95,  "fcos_s",      2, False),
]

LEVEL = 0

HEADER = """/* Auto-generated instruction codes. DO NOT EDIT. */
#ifndef PT_INST
#define PT_INST
"""
FOOTER = "\n#endif /* PT_INST */\n"


def encode_inst(num: int, nargs: int, branch: bool) -> str:
    b = int(branch)
    code = (LEVEL << 28) | ((nargs & 0xF) << 24) | ((num & 0xFF) << 16) | b
    return f"0x{code:08X}"


def generate(out_path: str):
    lines = [HEADER]
    seen = set()

    for num, name, nargs, branch in INSTRUCTIONS:
        if num in seen:
            raise SystemExit(f"错误：编号 {num} 重复")
        seen.add(num)
        if not (0 <= num <= 255):
            raise SystemExit(f"错误：编号 {num} 超出 8-bit 范围")
        if not (0 <= nargs <= 15):
            raise SystemExit(f"错误：{name} 参数个数 {nargs} 超出 4-bit 范围")

        code = encode_inst(num, nargs, branch)
        macro = f"INST_{name.upper()}"
        lines.append(f"#define {macro:30s} {code}")

    lines.append(FOOTER)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"生成完成：{len(INSTRUCTIONS)} 条指令 → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法：python3 gen_inst.py <输出路径>")
    generate(sys.argv[1])
