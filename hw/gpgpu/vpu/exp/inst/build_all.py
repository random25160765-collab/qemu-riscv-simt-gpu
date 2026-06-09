#!/usr/bin/env python3
"""
build_all.py — inst/ 层统一构建脚本

遍历所有 ISA 扩展, 调用 per-ISA gen_{isa}.py 生成 handler,
然后调用 gen_dispatch.py 聚合 dispatch 表。

用法:
    python3 build_all.py           # 全量生成
    python3 build_all.py --clean   # 清除所有生成产物
"""

import os, sys, subprocess
from glob import glob

INST_DIR = os.path.dirname(os.path.abspath(__file__))

# per-ISA 生成器 (相对 inst/ 目录)
ISA_GENERATORS = {
    "rv32i":     "rv32i/scripts/gen_rv32i.py",
    "rv32m":     "rv32m/scripts/gen_rv32m.py",
    "rv32f":     "rv32f/scripts/gen_rv32f.py",
    "rv32a":     "rv32a/scripts/gen_rv32a.py",
    "rv32zicsr": "rv32zicsr/scripts/gen_rv32zicsr.py",
    "sf":        "sf/scripts/gen_sf.py",
    "lp":        "lp/scripts/gen_lp.py",
    "rvv":       "rvv/scripts/gen_rvv.py",
}


def run(cmd, cwd=None):
    """Run a command and check return code."""
    print(f"  RUN  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  FAIL (exit {result.returncode})")
        return False
    return True


def build_all():
    """Run all generators."""
    print("=== build_all: generating handlers ===")
    ok = True

    for isa, script in ISA_GENERATORS.items():
        path = os.path.join(INST_DIR, script)
        if not os.path.exists(path):
            print(f"  SKIP {isa} — generator not found: {path}")
            continue
        if not run([sys.executable, path], cwd=os.path.dirname(path)):
            ok = False

    print("\n=== build_all: generating dispatch tables ===")
    disp_path = os.path.join(INST_DIR, "gen_dispatch.py")
    if not run([sys.executable, disp_path]):
        ok = False

    if ok:
        print("\nbuild_all: done.")
    else:
        print("\nbuild_all: completed with errors.")
        sys.exit(1)


def clean_all():
    """Remove all generated files."""
    print("=== build_all: cleaning generated files ===")

    # 各 ISA 的 handles/ 目录内容 (跳过手写 stub 的 mma)
    handles_dirs = glob(os.path.join(INST_DIR, "*/handles"))
    for d in handles_dirs:
        if "mma" in d:
            continue   # mma/handles/ 是手写 stub, 不清除
        for f in glob(os.path.join(d, "*.h")):
            os.remove(f)
            print(f"  RM  {os.path.relpath(f, INST_DIR)}")

    # dispatch 生成文件
    generated = [
        os.path.join(INST_DIR, "dispatch_list.h"),
        os.path.join(INST_DIR, "dispatch_rebind.h"),
    ]
    for f in generated:
        if os.path.exists(f):
            os.remove(f)
            print(f"  RM  {os.path.relpath(f, INST_DIR)}")

    print("build_all: clean done.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        clean_all()
    else:
        build_all()
