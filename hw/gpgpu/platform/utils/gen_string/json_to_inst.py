#!/usr/bin/env python3
"""
json_to_inst.py — JSON 指令 trace → GPGPU_INST 可读文本格式

用法: python3 json_to_inst.py <输入json文件> [输出文件]

参数映射说明:
  binary trace 只记录原始操作数，旧格式包含运行时计算值。
  本解析器尽可能从操作数重建计算值；无法重建的（内存 load 值、CSR old_val）则省略。
"""

import sys
import json
import struct


def s32(v):
    """将 uint32 解释为 int32。"""
    return struct.unpack("i", struct.pack("I", v & 0xFFFFFFFF))[0]


# ============================================================
# 每个 opcode 的格式化函数: (operands) -> str
# ============================================================

def _fmt_jal(op):
    # ops: [rd, imm, pc]  →  [JAL] rd=%d, imm=0x%x, pc=0x%x -> 0x%x
    rd, imm, pc = op[0], op[1], op[2]
    return f"\t\t[JAL] rd={rd}, imm=0x{imm:x}, pc=0x{pc:x} -> 0x{(pc + imm - 4) & 0xFFFFFFFF:x}"


def _fmt_jalr(op):
    # ops: [rd, src1, imm, pc]  →  [JALR] rd=%d, src1=0x%x, imm=0x%x, pc=0x%x -> 0x%x
    rd, src1, imm, pc = op[0], op[1], op[2], op[3]
    target = (src1 + imm) & ~1
    return f"\t\t[JALR] rd={rd}, src1=0x{src1:x}, imm=0x{imm:x}, pc=0x{pc:x} -> 0x{target:x}"


def _fmt_beq(op):
    # ops: [src1, src2, imm, pc]  →  taken = src1 == src2
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BEQ] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s1 == s2)}"


def _fmt_bne(op):
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BNE] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s1 != s2)}"


def _fmt_blt(op):
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BLT] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s32(s1) < s32(s2))}"


def _fmt_bge(op):
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BGE] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s32(s1) >= s32(s2))}"


def _fmt_bltu(op):
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BLTU] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s1 < s2)}"


def _fmt_bgeu(op):
    s1, s2, imm, pc = op[0], op[1], op[2], op[3]
    return f"\t\t[BGEU] src1=0x{s1:x}, src2=0x{s2:x}, imm=0x{imm:x}, pc=0x{pc:x}, taken={int(s1 >= s2)}"


def _fmt_lb(op):
    # ops: [rd, addr]  — val/sign-extended 不在 trace 中，省略
    return f"\t\t[LB] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_lh(op):
    return f"\t\t[LH] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_lw(op):
    return f"\t\t[LW] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_lbu(op):
    return f"\t\t[LBU] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_lhu(op):
    return f"\t\t[LHU] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_sb(op):
    # ops: [addr, src2]  →  val = src2 & 0xFF
    return f"\t\t[SB] addr=0x{op[0]:x}, val=0x{op[1] & 0xFF:02x}"


def _fmt_sh(op):
    return f"\t\t[SH] addr=0x{op[0]:x}, val=0x{op[1] & 0xFFFF:04x}"


def _fmt_sw(op):
    return f"\t\t[SW] addr=0x{op[0]:x}, val=0x{op[1]:08x}"


def _fmt_lui(op):
    # ops: [rd, imm]  →  result = imm
    return f"\t\t[LUI] rd={op[0]}, imm=0x{op[1]:x}, result=0x{op[1]:x}"


def _fmt_auipc(op):
    # ops: [rd, imm, pc]  →  result = pc + imm
    return f"\t\t[AUIPC] rd={op[0]}, imm=0x{op[1]:x}, pc=0x{op[2]:x}, result=0x{(op[1] + op[2]) & 0xFFFFFFFF:x}"


def _fmt_addi(op):
    # ops: [rd, src1, imm]
    return f"\t\t[ADDI] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result=0x{(op[1] + op[2]) & 0xFFFFFFFF:x}"


def _fmt_slti(op):
    return f"\t\t[SLTI] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result={int(s32(op[1]) < op[2])}"


def _fmt_sltiu(op):
    return f"\t\t[SLTIU] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result={int(op[1] < op[2])}"


def _fmt_xori(op):
    return f"\t\t[XORI] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result=0x{(op[1] ^ op[2]):x}"


def _fmt_ori(op):
    return f"\t\t[ORI] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result=0x{(op[1] | op[2]):x}"


def _fmt_andi(op):
    return f"\t\t[ANDI] rd={op[0]}, src1=0x{op[1]:x}, imm=0x{op[2]:x}, result=0x{(op[1] & op[2]):x}"


def _fmt_slli(op):
    # ops: [rd, src1, shamt]  — shamt already & 0x1F
    shamt = op[2]
    return f"\t\t[SLLI] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{(op[1] << shamt) & 0xFFFFFFFF:x}"


def _fmt_srli(op):
    shamt = op[2]
    return f"\t\t[SRLI] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{(op[1] >> shamt):x}"


def _fmt_srai(op):
    shamt = op[2]
    result = s32(op[1]) >> shamt
    return f"\t\t[SRAI] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_add(op):
    return f"\t\t[ADD] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{(op[1] + op[2]) & 0xFFFFFFFF:x}"


def _fmt_sub(op):
    return f"\t\t[SUB] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{(op[1] - op[2]) & 0xFFFFFFFF:x}"


def _fmt_sll(op):
    # ops: [rd, src1, shamt]  — shamt already & 0x1F
    shamt = op[2]
    return f"\t\t[SLL] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{(op[1] << shamt) & 0xFFFFFFFF:x}"


def _fmt_slt(op):
    return f"\t\t[SLT] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result={int(s32(op[1]) < s32(op[2]))}"


def _fmt_sltu(op):
    return f"\t\t[SLTU] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result={int(op[1] < op[2])}"


def _fmt_xor(op):
    return f"\t\t[XOR] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{(op[1] ^ op[2]):x}"


def _fmt_srl(op):
    shamt = op[2]
    return f"\t\t[SRL] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{(op[1] >> shamt):x}"


def _fmt_sra(op):
    shamt = op[2]
    result = s32(op[1]) >> shamt
    return f"\t\t[SRA] rd={op[0]}, src1=0x{op[1]:x}, shamt={shamt}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_or(op):
    return f"\t\t[OR] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{(op[1] | op[2]):x}"


def _fmt_and(op):
    return f"\t\t[AND] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{(op[1] & op[2]):x}"


def _fmt_mul(op):
    return f"\t\t[MUL] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{((op[1] & 0xFFFFFFFF) * (op[2] & 0xFFFFFFFF)) & 0xFFFFFFFF:x}"


def _fmt_mulh(op):
    a, b = s32(op[1]), s32(op[2])
    result = (a * b) >> 32
    return f"\t\t[MULH] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_mulhsu(op):
    a, b = s32(op[1]), op[2] & 0xFFFFFFFF
    result = (a * b) >> 32
    return f"\t\t[MULHSU] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_mulhu(op):
    a, b = op[1] & 0xFFFFFFFF, op[2] & 0xFFFFFFFF
    result = (a * b) >> 32
    return f"\t\t[MULHU] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_div(op):
    a, b = s32(op[1]), s32(op[2])
    result = -1 if b == 0 else a // b
    return f"\t\t[DIV] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_divu(op):
    a, b = op[1] & 0xFFFFFFFF, op[2] & 0xFFFFFFFF
    result = 0xFFFFFFFF if b == 0 else a // b
    return f"\t\t[DIVU] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_rem(op):
    a, b = s32(op[1]), s32(op[2])
    result = a if b == 0 else a % b
    return f"\t\t[REM] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_remu(op):
    a, b = op[1] & 0xFFFFFFFF, op[2] & 0xFFFFFFFF
    result = a if b == 0 else a % b
    return f"\t\t[REMU] rd={op[0]}, src1=0x{op[1]:x}, src2=0x{op[2]:x}, result=0x{result & 0xFFFFFFFF:x}"


def _fmt_csrrw(op):
    # ops: [rd, csr, src1]  — old_val/new_val 不在 trace 中，省略
    return f"\t\t[CSRRW] rd={op[0]}, csr=0x{op[1]:x}, src1=0x{op[2]:x}"


def _fmt_csrrs(op):
    return f"\t\t[CSRRS] rd={op[0]}, csr=0x{op[1]:x}, src1=0x{op[2]:x}"


def _fmt_csrrc(op):
    return f"\t\t[CSRRC] rd={op[0]}, csr=0x{op[1]:x}, src1=0x{op[2]:x}"


def _fmt_csrrwi(op):
    # ops: [rd, csr, uimm]
    return f"\t\t[CSRRWI] rd={op[0]}, csr=0x{op[1]:x}, uimm={op[2]}"


def _fmt_csrrsi(op):
    return f"\t\t[CSRRSI] rd={op[0]}, csr=0x{op[1]:x}, uimm={op[2]}"


def _fmt_csrrci(op):
    # ops: [rd, csr, imm]  — old_val/new_val 不在 trace 中
    return f"\t\t[CSRRCI] rd={op[0]}, csr=0x{op[1]:x}, imm={op[2]}"


def _fmt_ecall(op):
    return "\t\t[ECALL]"


def _fmt_ebreak(op):
    return "\t\t[EBREAK] breakpoint instruction executed"


def _fmt_flw(op):
    return f"\t\t[FLW] rd={op[0]}, addr=0x{op[1]:x}"


def _fmt_fsw(op):
    return f"\t\t[FSW] addr=0x{op[0]:x}, val=0x{op[1]:08x}"


def _fmt_fmadd_s(op):
    return f"\t\t[FMADD_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}, rs3=0x{op[3]:08x}"


def _fmt_fmsub_s(op):
    return f"\t\t[FMSUB_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}, rs3=0x{op[3]:08x}"


def _fmt_fnmsub_s(op):
    return f"\t\t[FNMSUB_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}, rs3=0x{op[3]:08x}"


def _fmt_fnmadd_s(op):
    return f"\t\t[FNMADD_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}, rs3=0x{op[3]:08x}"


def _fmt_fadd_s(op):
    return f"\t\t[FADD_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fsub_s(op):
    return f"\t\t[FSUB_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fmul_s(op):
    return f"\t\t[FMUL_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fdiv_s(op):
    return f"\t\t[FDIV_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fsqrt_s(op):
    return f"\t\t[FSQRT_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fsgnj_s(op):
    return f"\t\t[FSGNJ_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fsgnjn_s(op):
    return f"\t\t[FSGNJN_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fsgnjx_s(op):
    return f"\t\t[FSGNJX_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fmin_s(op):
    return f"\t\t[FMIN_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fmax_s(op):
    return f"\t\t[FMAX_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fcvt_w_s(op):
    return f"\t\t[FCVT_W_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_wu_s(op):
    return f"\t\t[FCVT_WU_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fmv_x_w(op):
    return f"\t\t[FMV_X_W] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_feq_s(op):
    return f"\t\t[FEQ_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_flt_s(op):
    return f"\t\t[FLT_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fle_s(op):
    return f"\t\t[FLE_S] rd={op[0]}, rs1=0x{op[1]:08x}, rs2=0x{op[2]:08x}"


def _fmt_fclass_s(op):
    return f"\t\t[FCLASS_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_w(op):
    return f"\t\t[FCVT_S_W] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_wu(op):
    return f"\t\t[FCVT_S_WU] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fmv_w_x(op):
    return f"\t\t[FMV_W_X] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_bf16(op):
    return f"\t\t[FCVT_S_BF16] rd={op[0]}, rs1=0x{op[1] & 0xFFFF:04x}"


def _fmt_fcvt_bf16_s(op):
    return f"\t\t[FCVT_BF16_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_e4m3(op):
    return f"\t\t[FCVT_S_E4M3] rd={op[0]}, rs1=0x{op[1] & 0xFF:02x}"


def _fmt_fcvt_e4m3_s(op):
    return f"\t\t[FCVT_E4M3_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_e5m2(op):
    return f"\t\t[FCVT_S_E5M2] rd={op[0]}, rs1=0x{op[1] & 0xFF:02x}"


def _fmt_fcvt_e5m2_s(op):
    return f"\t\t[FCVT_E5M2_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcvt_s_e2m1(op):
    return f"\t\t[FCVT_S_E2M1] rd={op[0]}, rs1=0x{op[1] & 0xFF:02x}"


def _fmt_fcvt_e2m1_s(op):
    return f"\t\t[FCVT_E2M1_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fexp_s(op):
    return f"\t\t[FEXP_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fln_s(op):
    return f"\t\t[FLN_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_frcp_s(op):
    return f"\t\t[FRCP_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_frsqrt_s(op):
    return f"\t\t[FRSQRT_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_ftanh_s(op):
    return f"\t\t[FTANH_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fsigmoid_s(op):
    return f"\t\t[FSIGMOID_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fsin_s(op):
    return f"\t\t[FSIN_S] rd={op[0]}, rs1=0x{op[1]:08x}"


def _fmt_fcos_s(op):
    return f"\t\t[FCOS_S] rd={op[0]}, rs1=0x{op[1]:08x}"


# ============================================================
# opcode → 格式化函数 映射表
# ============================================================

FMT_TABLE = {
    1:  _fmt_jal,
    2:  _fmt_jalr,
    3:  _fmt_beq,
    4:  _fmt_bne,
    5:  _fmt_blt,
    6:  _fmt_bge,
    7:  _fmt_bltu,
    8:  _fmt_bgeu,
    9:  _fmt_lb,
    10: _fmt_lh,
    11: _fmt_lw,
    12: _fmt_lbu,
    13: _fmt_lhu,
    14: _fmt_sb,
    15: _fmt_sh,
    16: _fmt_sw,
    17: _fmt_lui,
    18: _fmt_auipc,
    19: _fmt_addi,
    20: _fmt_slti,
    21: _fmt_sltiu,
    22: _fmt_xori,
    23: _fmt_ori,
    24: _fmt_andi,
    25: _fmt_slli,
    26: _fmt_srli,
    27: _fmt_srai,
    28: _fmt_add,
    29: _fmt_sub,
    30: _fmt_sll,
    31: _fmt_slt,
    32: _fmt_sltu,
    33: _fmt_xor,
    34: _fmt_srl,
    35: _fmt_sra,
    36: _fmt_or,
    37: _fmt_and,
    38: _fmt_mul,
    39: _fmt_mulh,
    40: _fmt_mulhsu,
    41: _fmt_mulhu,
    42: _fmt_div,
    43: _fmt_divu,
    44: _fmt_rem,
    45: _fmt_remu,
    46: _fmt_csrrw,
    47: _fmt_csrrs,
    48: _fmt_csrrc,
    49: _fmt_csrrwi,
    50: _fmt_csrrsi,
    51: _fmt_csrrci,
    52: _fmt_ecall,
    53: _fmt_ebreak,
    54: _fmt_flw,
    55: _fmt_fsw,
    56: _fmt_fmadd_s,
    57: _fmt_fmsub_s,
    58: _fmt_fnmsub_s,
    59: _fmt_fnmadd_s,
    60: _fmt_fadd_s,
    61: _fmt_fsub_s,
    62: _fmt_fmul_s,
    63: _fmt_fdiv_s,
    64: _fmt_fsqrt_s,
    65: _fmt_fsgnj_s,
    66: _fmt_fsgnjn_s,
    67: _fmt_fsgnjx_s,
    68: _fmt_fmin_s,
    69: _fmt_fmax_s,
    70: _fmt_fcvt_w_s,
    71: _fmt_fcvt_wu_s,
    72: _fmt_fmv_x_w,
    73: _fmt_feq_s,
    74: _fmt_flt_s,
    75: _fmt_fle_s,
    76: _fmt_fclass_s,
    77: _fmt_fcvt_s_w,
    78: _fmt_fcvt_s_wu,
    79: _fmt_fmv_w_x,
    80: _fmt_fcvt_s_bf16,
    81: _fmt_fcvt_bf16_s,
    82: _fmt_fcvt_s_e4m3,
    83: _fmt_fcvt_e4m3_s,
    84: _fmt_fcvt_s_e5m2,
    85: _fmt_fcvt_e5m2_s,
    86: _fmt_fcvt_s_e2m1,
    87: _fmt_fcvt_e2m1_s,
    88: _fmt_fexp_s,
    89: _fmt_fln_s,
    90: _fmt_frcp_s,
    91: _fmt_frsqrt_s,
    92: _fmt_ftanh_s,
    93: _fmt_fsigmoid_s,
    94: _fmt_fsin_s,
    95: _fmt_fcos_s,
}


def json_to_inst(records):
    """将 JSON 记录列表转为 GPGPU_INST 可读文本列表。"""
    lines = []
    for rec in records:
        fields = rec["fields"]
        operands = rec["operands"]
        opcode = fields["opcode"]

        fmt_fn = FMT_TABLE.get(opcode)
        if fmt_fn:
            lines.append(fmt_fn(operands))
        else:
            args_str = ", ".join(f"0x{op:x}" if isinstance(op, int) else str(op) for op in operands)
            lines.append(f"\t\t[UNKNOWN_OPCODE_{opcode}] {args_str}")

    return lines


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 json_to_inst.py <输入json文件> [输出文件]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(in_path, "r") as f:
        records = json.load(f)

    lines = json_to_inst(records)
    output = "\n".join(lines) + "\n"

    if out_path:
        with open(out_path, "w") as f:
            f.write(output)
        print(f"转换完成：{len(lines)} 条指令 → {out_path}")
    else:
        print(output)