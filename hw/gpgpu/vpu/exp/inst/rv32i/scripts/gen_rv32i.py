#!/usr/bin/env python3
"""
gen_rv32i.py — RV32I handler 生成器

从 rv32i_spec.yaml 生成每个指令的 computed-goto handler 代码,
供 engine.c 通过 #include 引入。

用法:
    python3 gen_rv32i.py [spec.yaml]

设计:
    - 每条标量指令生成一个 op_xxx: label + 代码块
    - 所有 handler 共享 engine_exec 作用域的隐式接口
    - 生成文件放在 handles/ 目录
"""

import yaml
import os
import sys
from textwrap import dedent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = SCRIPT_DIR
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "handles")


def gen_alu_imm(inst):
    """rd = rs1 OP (uint32_t)imm"""
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = GPR(rs1, _li) {inst['op']} (uint32_t)imm;
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_slti_imm(inst):
    """rd = (rs1 < imm) ? 1 : 0 with signed/unsigned"""
    if inst["signed"]:
        cmp = "((int32_t)GPR(rs1, _li) < imm)"
    else:
        cmp = "(GPR(rs1, _li) < (uint32_t)imm)"
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = {cmp} ? 1 : 0;
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_shift_imm(inst):
    """rd = rs1 OP (imm & 0x1F), with sign handling for srai"""
    if inst["signed"]:
        rhs = "(uint32_t)((int32_t)GPR(rs1, _li) >> (imm & 0x1F))"
    else:
        rhs = "GPR(rs1, _li) {op} (imm & 0x1F)".format(op=inst["op"])
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = {rhs};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_alu_reg(inst):
    """rd = rs1 OP rs2"""
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = GPR(rs1, _li) {inst['op']} GPR(rs2, _li);
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_shift_reg(inst):
    """rd = rs1 OP (rs2 & 0x1F), with sign handling for sra"""
    if inst["signed"]:
        rhs = "(uint32_t)((int32_t)GPR(rs1, _li) >> (GPR(rs2, _li) & 0x1F))"
    else:
        rhs = "GPR(rs1, _li) {op} (GPR(rs2, _li) & 0x1F)".format(op=inst["op"])
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = {rhs};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_slt_reg(inst):
    """rd = (rs1 < rs2) ? 1 : 0"""
    if inst["signed"]:
        cmp = "((int32_t)GPR(rs1, _li) < (int32_t)GPR(rs2, _li))"
    else:
        cmp = "(GPR(rs1, _li) < GPR(rs2, _li))"
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            GPR(rd, _li) = {cmp} ? 1 : 0;
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_jump(inst):
    """jal / jalr"""
    if inst["type"] == "jal":
        return dedent("""\
        op_jal:
        {
            int rd = ip[-1].rd, imm = ip[-1].imm;
            FOR_EACH_LANE
            {
                GPR(rd, _li) = PC(_li) + 4;
                PC(_li) += imm;
            }
            ip = &code[ip[-1].branch_tgt];
            NEXT();
        }
        """)
    else:  # jalr
        return dedent("""\
        op_jalr:
        {
            int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
            int32_t target = (int32_t)(GPR(rs1, 0) + imm) & ~1;
            FOR_EACH_LANE
            {
                GPR(rd, _li) = PC(_li) + 4;
                PC(_li) = (uint32_t)target;
            }
            ip = &code[(target - s->kernel.kernel_addr) / 4];
            NEXT();
        }
        """)


def gen_upper_imm(inst):
    """lui / auipc"""
    if inst["type"] == "lui":
        return dedent("""\
        op_lui:
        {
            int rd = ip[-1].rd, imm = ip[-1].imm;
            FOR_EACH_LANE
            {
                GPR(rd, _li) = (uint32_t)imm;
                PC(_li) += 4;
            }
            NEXT();
        }
        """)
    else:  # auipc
        return dedent("""\
        op_auipc:
        {
            int rd = ip[-1].rd, imm = ip[-1].imm;
            FOR_EACH_LANE
            {
                uint32_t instr_pc = s->kernel.kernel_addr + (uint32_t)(ip - code - 1) * 4;
                GPR(rd, _li) = instr_pc + imm;
                PC(_li) += 4;
            }
            NEXT();
        }
        """)


def gen_branch(inst):
    """DIV_BR macro call"""
    return "op_{name}:\n    DIV_BR({cond});".format(
        name=inst["name"], cond=inst["cond"]
    )


def gen_load(inst):
    """per-lane load with sign/zero extension"""
    name = inst["name"]
    width = inst["width"]
    use_ctrl = inst.get("use_ctrl", False)
    ext = inst.get("extend", "none")

    if use_ctrl:
        # lw uses ctrl_read (handles VRAM + shared memory + CTRL regs)
        body = "GPR(rd, _li) = ctrl_read(ctx, a, _li);"
    elif width == 1 and ext == "sext":
        body = "uint8_t v = (uint8_t)gpu_read(s, a, 1);\n"
        body += "            " + "GPR(rd, _li) = (uint32_t)((int32_t)(v << 24) >> 24);"
    elif width == 2 and ext == "sext":
        body = "uint16_t v = (uint16_t)gpu_read(s, a, 2);\n"
        body += "            " + "GPR(rd, _li) = (uint32_t)((int32_t)(v << 16) >> 16);"
    elif ext == "zext":
        body = "GPR(rd, _li) = (uint32_t)gpu_read(s, a, {w});".format(w=width)
    else:
        body = "GPR(rd, _li) = (uint32_t)gpu_read(s, a, {w});".format(w=width)

    return dedent(f"""\
    op_{name}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
        FOR_EACH_LANE
        {{
            uint32_t a = GPR(rs1, _li) + imm;
            {body}
            PC(_li) += 4;
        }}
        PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * {width};)
        COALESCE(ip[-1].rs1, ip[-1].imm);
        NEXT();
    }}
    """)


def gen_store(inst):
    """per-lane store"""
    name = inst["name"]
    width = inst["width"]
    return dedent(f"""\
    op_{name}:
    {{
        int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
        FOR_EACH_LANE
        {{
            gpu_write(s, GPR(rs1, _li) + imm, {width}, GPR(rs2, _li));
            PC(_li) += 4;
        }}
        PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * {width};)
        NEXT();
    }}
    """)


# ============================================================================
# Template dispatch table
# ============================================================================

GEN_FUNCTIONS = {
    "alu_imm":    gen_alu_imm,
    "slti_imm":   gen_slti_imm,
    "shift_imm":  gen_shift_imm,
    "alu_reg":    gen_alu_reg,
    "shift_reg":  gen_shift_reg,
    "slt_reg":    gen_slt_reg,
    "jump":       gen_jump,
    "upper_imm":  gen_upper_imm,
    "branch":     gen_branch,
    "load":       gen_load,
    "store":      gen_store,
}


def header(family_name, desc):
    return dedent(f"""\
    /*
     * rv32i_{family_name}.h — {desc}
     * Auto-generated by gen_rv32i.py — DO NOT EDIT
     */
    """)


def generate(spec_path):
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for family in spec["families"]:
        name = family["name"]
        template = family["template"]
        desc = family.get("desc", name)
        gen_fn = GEN_FUNCTIONS[template]

        output = header(name, desc)
        for inst in family["instructions"]:
            output += gen_fn(inst) + "\n"

        fname = os.path.join(OUT_DIR, f"rv32i_{name}.h")
        with open(fname, "w") as f:
            f.write(output)
        print(f"  GEN  rv32i_{name}.h")

    # Generate all-include
    lines = [
        "/* rv32i_all.h — 汇总所有 RV32I handler",
        " * Auto-generated by gen_rv32i.py — DO NOT EDIT",
        " */",
        "",
    ]
    for family in spec["families"]:
        lines.append(f'#include "rv32i_{family["name"]}.h"')
    lines.append("")

    fname = os.path.join(OUT_DIR, "rv32i_all.h")
    with open(fname, "w") as f:
        f.write("\n".join(lines))
    print(f"  GEN  rv32i_all.h")

    return len(spec["families"])


if __name__ == "__main__":
    spec_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SPEC_DIR, "rv32i_spec.yaml")
    n = generate(spec_path)
    total_inst = sum(1 for f in yaml.safe_load(open(spec_path))["families"]
                     for _ in f["instructions"])
    print(f"\nDone. {n} families, {total_inst} handlers → handles/")
