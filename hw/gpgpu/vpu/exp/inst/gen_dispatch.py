#!/usr/bin/env python3
"""
gen_dispatch.py — 指令解码分发表生成器 (dispatch trie 元信息)

从所有启用的 ISA spec 收集每条指令的 decode 信息 (pattern, type, imm_fn),
生成:
  1. inst/dispatch_list.h     — INSTRUCTION_LIST + NUM_OF_INST + DISP_ enum
  2. inst/dispatch_rebind.h   — rvv_dispatch_rebind() 热替换函数

核心原则:
  - 一个 bit pattern = 一个 trie 条目 = 一个 dispatch slot
  - RVV 的 SEW 变体 (e8/e16/e32) 共享同一 pattern，不在 INSTRUCTION_LIST 里重复
  - vsetvli 时通过 dispatch_rebind.h 热替换 dispatch[slot] 指向的 handler
  - DISP_ enum 给每个 slot 一个编译期常量名，rebind 函数不用魔法数字

用法:
    python3 gen_dispatch.py                        # 所有 ISA
    python3 gen_dispatch.py --isa rv32i,rv32m,rvv  # 指定 ISA

YAML spec decode 字段:
    标量指令 (无变体):
      - { name: addi, op: "+",
          decode: { pattern: "...", type: TYPE_I, imm: immI } }

    RVV 指令 (有 SEW 变体, decode.variants 描述 rebind 信息):
      - { name: vadd, op: "+", itype: int,
          decode: { pattern: "000000? ????? ????? 000 ????? 10101 11",
                    type: TYPE_R, imm: imm0,
                    variants: { base: "vadd_vv",       # handler 命名基础
                                sew: [8, 16, 32] } } }  # 已实现的 SEW

    对于 RVV, 每个 form (.vv/.vx/.vi/.vf) 有独立的 pattern → 独立的 trie 条目
    → 独立的 DISP_VADD_VV / DISP_VADD_VX / DISP_VADD_VI 枚举值
"""

import yaml
import os
import sys
from glob import glob
from collections import OrderedDict
from textwrap import dedent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR   # 生成到 inst/ 目录


# ============================================================================
# 扫描
# ============================================================================

def scan_specs(enabled_isas=None):
    """扫描 inst/*/scripts/*.yaml"""
    specs = []
    pattern = os.path.join(SCRIPT_DIR, "*/scripts/*.yaml")
    for fname in sorted(glob(pattern)):
        isa = os.path.basename(os.path.dirname(os.path.dirname(fname)))
        if enabled_isas and isa not in enabled_isas:
            print(f"  SKIP {isa} (not enabled)")
            continue
        with open(fname) as f:
            spec = yaml.safe_load(f)
        specs.append((isa, spec, fname))
    return specs


# ============================================================================
# 收集
# ============================================================================

def collect_decode_entries(specs):
    """
    返回 (trie_entries, variant_entries):
      trie_entries:     [{name, label, pattern, type, imm, isa}, ...]
                        每个 bit pattern 一条 → INSTRUCTION_LIST
      variant_entries:  [{label, base, sew_list, isa}, ...]
                        有 SEW 变体的条目 → dispatch_rebind 使用
    """
    trie = []
    variants = []
    seen = set()
    skipped = []

    def _collect(instructions, context):
        for inst in instructions:
            decode = inst.get("decode")
            if decode is None:
                skipped.append(f"{inst['name']} ({context})")
                continue

            label = decode.get("label", f"op_{inst['name']}")

            # dispatch_name: label 去掉 "op_" 前缀 → engine.c 做 &&op_##name
            if label.startswith("op_"):
                dispatch_name = label[3:]
            else:
                dispatch_name = inst["name"]

            # 去重
            key = (label, decode["pattern"])
            if key in seen:
                print(f"  WARN: duplicate {label} — skipping")
                continue
            seen.add(key)

            entry = {
                "name": dispatch_name,
                "label": label,
                "pattern": decode["pattern"],
                "type": decode["type"],
                "imm": decode["imm"],
                "isa": isa,
            }
            trie.append(entry)

            # 收集 SEW 变体信息
            var_info = decode.get("variants")
            if var_info:
                entry["variants"] = var_info  # 保留给 rebind 用
                variants.append({
                    "label": label,
                    "base": var_info.get("base", dispatch_name),
                    "sew_list": var_info.get("sew", [32]),
                    "prefix": var_info.get("prefix", "rvv_"),
                    "isa": isa,
                })

    for isa, spec, fname in specs:
        for family in spec.get("families", []):
            _collect(family.get("instructions", []),
                     f"family:{family['name']} ({isa})")
        for misc_entry in spec.get("misc", []):
            _collect([misc_entry], f"misc ({isa})")

    if skipped:
        print(f"\n  WARN: {len(skipped)} instructions missing decode info:")
        for s in skipped[:10]:
            print(f"    - {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return trie, variants


# ============================================================================
# 生成: dispatch_list.h
# ============================================================================

def generate_dispatch_list(trie_entries):
    """生成 INSTRUCTION_LIST + NUM_OF_INST + DISP_ enum"""

    grouped = OrderedDict()
    for e in trie_entries:
        grouped.setdefault(e["isa"], []).append(e)

    lines = [
        "/*",
        " * dispatch_list.h — 指令解码分发表",
        " * Auto-generated by gen_dispatch.py — DO NOT EDIT",
        " */",
        "#ifndef INST_DISPATCH_LIST_H",
        "#define INST_DISPATCH_LIST_H",
        "",
        "/*",
        " * DISP_ enum — 每条指令在 dispatch[] 中的索引 (编译期常量)",
        " * trie 匹配 bit pattern → 返回 DISP_xxx → dispatch[DISP_xxx] = handler",
        " * RVV rebind:  dispatch[DISP_VADD_VV] = &&rvv_vadd_vv_e8",
        " */",
        "enum {",
    ]

    idx = 0
    for isa, group in grouped.items():
        lines.append(f"    /* ── {isa} ── */")
        for e in group:
            # label 去掉 op_ 前缀做枚举名
            enum_name = e["label"].replace("op_", "DISP_").upper()
            lines.append(f"    {enum_name} = {idx},")
            idx += 1

    lines.append(f"    DISP_COUNT = {idx}")
    lines.append("};")
    lines.append("")

    # INSTRUCTION_LIST — 保持和原来 dispatch.h 一样的格式
    lines.append("/*")
    lines.append(" * INSTRUCTION_LIST — X(name, pattern, op_type, imm_fn)")
    lines.append(" * trie 构造引擎用此遍历所有 bit pattern")
    lines.append(" */")
    lines.append("#define INSTRUCTION_LIST \\")

    for isa, group in grouped.items():
        lines.append(f"    /* ── {isa} ── */ \\")
        for e in group:
            lines.append(
                f'    X({e["name"]}, "{e["pattern"]}", {e["type"]}, {e["imm"]}); \\'
            )

    lines.append("    /* end */")
    lines.append("")
    lines.append(f"#define NUM_OF_INST {len(trie_entries)}")
    lines.append("")
    lines.append("#endif /* INST_DISPATCH_LIST_H */")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 生成: dispatch_rebind.h
# ============================================================================

def generate_dispatch_rebind(variant_entries):
    """生成 rvv_dispatch_rebind() — vsetvli 调用, 热替换 dispatch 表"""

    if not variant_entries:
        return dedent("""\
        /* dispatch_rebind.h — no SEW-variable instructions */
        #define rvv_dispatch_rebind(dispatch, sew) ((void)(dispatch), (void)(sew))
        """)

    # 收集所有涉及的 SEW
    all_sew = sorted(set(
        s for v in variant_entries for s in v["sew_list"]
    ))

    lines = [
        "/*",
        " * dispatch_rebind.h — vsetvli dispatch 热替换",
        " * Auto-generated by gen_dispatch.py — DO NOT EDIT",
        " *",
        " * vsetvli 调用 rvv_dispatch_rebind(dispatch, sew)：",
        " *   把 dispatch[] 中所有 RVV 条目重定向到对应 SEW 的 handler",
        " *",
        " *   例: dispatch[DISP_VADD_VV] = &&rvv_vadd_vv_e8;",
        " */",
        "",
        "/* 必须在 engine_exec 函数体内 #include (依赖 &&rvv_* computed-goto 标签) */",
        "#define RVV_DISPATCH_REBIND(disp, sew) do { \\",
        "    switch (sew) { \\",
    ]

    for sew in all_sew:
        lines.append(f"    case {sew}: \\")
        for v in variant_entries:
            if sew not in v["sew_list"]:
                continue
            base = v["base"]
            prefix = v.get("prefix", "rvv_")
            enum_name = v["label"].replace("op_", "DISP_").upper()
            lines.append(
                f"        disp[{enum_name}] = &&{prefix}{base}_e{sew}; \\"
            )
        lines.append("        break; \\")

    lines.append("    default: \\")
    lines.append('        fprintf(stderr, "RVV_DISPATCH_REBIND: unsupported SEW=%d\\n", sew); \\')
    lines.append("        break; \\")
    lines.append("    } \\")
    lines.append("} while(0)")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 入口
# ============================================================================

def parse_args():
    enabled = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--isa" and i + 1 < len(args):
            enabled = set(args[i + 1].split(","))
            i += 2
        else:
            i += 1
    return enabled


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    enabled_isas = parse_args()
    print(f"gen_dispatch: {'all' if not enabled_isas else sorted(enabled_isas)}")

    specs = scan_specs(enabled_isas)
    print(f"  found {len(specs)} spec(s): {[s[0] for s in specs]}")

    trie, variants = collect_decode_entries(specs)
    print(f"  collected {len(trie)} decode entries, "
          f"{len(variants)} with SEW variants")

    if not trie:
        print("  ERROR: no instructions with decode info")
        sys.exit(1)

    # 产物 1: dispatch_list.h
    out = os.path.join(OUT_DIR, "dispatch_list.h")
    with open(out, "w") as f:
        f.write(generate_dispatch_list(trie))
    print(f"  GEN  {os.path.relpath(out)}")

    # 产物 2: dispatch_rebind.h
    out = os.path.join(OUT_DIR, "dispatch_rebind.h")
    with open(out, "w") as f:
        f.write(generate_dispatch_rebind(variants))
    print(f"  GEN  {os.path.relpath(out)}")

    n_var = sum(1 + len(v["sew_list"]) for v in variants)
    print(f"\nDone. {len(trie)} trie entries + {n_var} SEW-variant handlers")
    print(f"  NUM_OF_INST = {len(trie)}")
