#!/usr/bin/env python3
"""gen_rv32f.py — RV32F 单精度浮点 handler 生成器"""

import yaml, os, sys
from textwrap import dedent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "handles")

# 复杂 handler 的硬编码模板 (太复杂不适合作参数化)
FCVT_W_S_BODY = """\
float f = FR(rs1, _li);
            uint32_t raw = FPR(rs1, _li);
            GPR(rd, _li) = (uint32_t)(int32_t)(((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF)
                                                       ? 0x7FFFFFFF
                                                       : (f >= 2147483648.0f ? 0x7FFFFFFF
                                                                             : (f < -2147483648.0f ? (int32_t)0x80000000
                                                                                                   : (int32_t)f)));"""

FCVT_WU_S_BODY = """\
float f = FR(rs1, _li);
            uint32_t raw = FPR(rs1, _li);
            GPR(rd, _li) = ((raw >> 23) & 0xFF) == 0xFF && (raw & 0x7FFFFF)
                                   ? 0xFFFFFFFF
                                   : (f < 0.0f ? 0 : (f >= 4294967296.0f ? 0xFFFFFFFF : (uint32_t)f));"""


def gen_float_bin(inst):
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            FR(rd, _li) = FR(rs1, _li) {inst['op']} FR(rs2, _li);
        }}
        FOR_EACH_LANE PC(_li) += 4;
        NEXT();
    }}
    """)


def gen_float_fma(inst):
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, rs3 = ip[-1].rs3;
        FOR_EACH_LANE
        {{
            FR(rd, _li) = ({inst['expr']});
        }}
        FOR_EACH_LANE PC(_li) += 4;
        NEXT();
    }}
    """)


def gen_float_sqrt(inst):
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE
        {{
            FR(rd, _li) = {inst['expr']};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_sgnj(inst):
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            FPR(rd, _li) = {inst['expr']};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_minmax(inst):
    cmp = inst["cmp"]
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            float a = FR(rs1, _li), b = FR(rs2, _li);
            FR(rd, _li) = (a != a) ? b : (b != b) ? a : (a {cmp} b ? a : b);
            if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_cmp(inst):
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
        FOR_EACH_LANE
        {{
            float a = FR(rs1, _li), b = FR(rs2, _li);
            GPR(rd, _li) = (a {inst['op']} b) ? 1 : 0;
            if (UNLIKELY((a != a) || (b != b))) _fcsr[_li] |= 0x10; /* NV */
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_fcvt(inst):
    name = inst["name"]
    if name == "fcvt_s_w":
        val = "(float)(int32_t)GPR(rs1, _li)"
        lhs = "FR(rd, _li)"
    elif name == "fcvt_s_wu":
        val = "(float)GPR(rs1, _li)"
        lhs = "FR(rd, _li)"
    elif name == "fcvt_w_s":
        val = FCVT_W_S_BODY
        # body already assigns to GPR
        return dedent(f"""\
        op_fcvt_w_s:
        {{
            int rd = ip[-1].rd, rs1 = ip[-1].rs1;
            FOR_EACH_LANE
            {{
                {val}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)
    elif name == "fcvt_wu_s":
        val = FCVT_WU_S_BODY
        return dedent(f"""\
        op_fcvt_wu_s:
        {{
            int rd = ip[-1].rd, rs1 = ip[-1].rs1;
            FOR_EACH_LANE
            {{
                {val}
                PC(_li) += 4;
            }}
            NEXT();
        }}
        """)
    else:
        val = "0"
        lhs = "GPR(rd, _li)"

    return dedent(f"""\
    op_{name}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE
        {{
            {lhs} = {val};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_mv(inst):
    expr = inst["expr"]
    return dedent(f"""\
    op_{inst['name']}:
    {{
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE
        {{
            {expr};
            PC(_li) += 4;
        }}
        NEXT();
    }}
    """)


def gen_float_fclass(inst):
    return dedent("""\
    op_fclass_s:
    {
        int rd = ip[-1].rd, rs1 = ip[-1].rs1;
        FOR_EACH_LANE
        {
            uint32_t b = FPR(rs1, _li), e = (b >> 23) & 0xFF, m = b & 0x7FFFFF, sgn = (b >> 31) & 1;
            int r = 0;
            if (e == 0xFF)
                r = m ? (sgn ? (1 << 9) : (1 << 8)) : (sgn ? (1 << 0) : (1 << 7));
            else if (e == 0)
                r = m ? (sgn ? (1 << 2) : (1 << 5)) : (sgn ? (1 << 3) : (1 << 4));
            else
                r = sgn ? (1 << 1) : (1 << 6);
            GPR(rd, _li) = (uint32_t)r;
            PC(_li) += 4;
        }
        NEXT();
    }
    """)


def gen_float_ldst(inst):
    if inst["type"] == "load":
        return dedent("""\
        op_flw:
        {
            int rd = ip[-1].rd, rs1 = ip[-1].rs1, imm = ip[-1].imm;
            FOR_EACH_LANE
            {
                uint32_t a = GPR(rs1, _li) + imm;
                FPR(rd, _li) = (LIKELY(a + 4 <= s->vram_size)) ? *(uint32_t *)(s->vram_ptr + a) : 0;
                PC(_li) += 4;
            }
            PERF_IF(s->stats.bytes_read += __builtin_popcount(_active) * 4;)
            COALESCE(ip[-1].rs1, ip[-1].imm);
            NEXT();
        }
        """)
    else:
        return dedent("""\
        op_fsw:
        {
            int rs1 = ip[-1].rs1, rs2 = ip[-1].rs2, imm = ip[-1].imm;
            FOR_EACH_LANE
            {
                uint32_t a = GPR(rs1, _li) + imm;
                if (LIKELY(a + 4 <= s->vram_size)) *(uint32_t *)(s->vram_ptr + a) = FPR(rs2, _li);
            }
            FOR_EACH_LANE PC(_li) += 4;
            PERF_IF(s->stats.bytes_write += __builtin_popcount(_active) * 4;)
            NEXT();
        }
        """)


GEN_FN = {
    "float_bin": gen_float_bin, "float_fma": gen_float_fma,
    "float_sqrt": gen_float_sqrt, "float_sgnj": gen_float_sgnj,
    "float_minmax": gen_float_minmax, "float_cmp": gen_float_cmp,
    "float_fcvt": gen_float_fcvt, "float_mv": gen_float_mv,
    "float_fclass": gen_float_fclass, "float_ldst": gen_float_ldst,
}


def generate(spec_path):
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    for family in spec["families"]:
        name = family["name"]
        output = f"/* rv32f_{name}.h — {family.get('desc', name)}\n * Auto-generated by gen_rv32f.py — DO NOT EDIT\n */\n"
        gen_fn = GEN_FN[family["template"]]
        for inst in family["instructions"]:
            output += gen_fn(inst) + "\n"
        fname = os.path.join(OUT_DIR, f"rv32f_{name}.h")
        with open(fname, "w") as f:
            f.write(output)
        print(f"  GEN  rv32f_{name}.h")
    all_lines = ["/* rv32f_all.h — Auto-generated */", ""]
    for fam in spec["families"]:
        all_lines.append(f'#include "rv32f_{fam["name"]}.h"')
    all_lines.append("")
    with open(os.path.join(OUT_DIR, "rv32f_all.h"), "w") as f:
        f.write("\n".join(all_lines))
    print(f"  GEN  rv32f_all.h")


if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "rv32f_spec.yaml")
    generate(sp)
