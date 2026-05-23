#!/usr/bin/env python3
"""
控制事件编码生成器 (level=0x01)
用法：python3 gen_event.py <输出路径>
"""

import sys

EVENTS = [
    # (编号, 名字, 参数个数, 保留)
    (0x01, "reg_write",       2, False),
    (0x02, "reg_read",        2, False),
    (0x03, "dma_start",       5, False),
    (0x04, "dma_complete",    1, False),
    (0x05, "kernel_dispatch", 11, False),
    (0x06, "kernel_complete", 1, False),
    (0x07, "irq_fire",        2, False),
    (0x08, "error_event",     2, False),
    (0x09, "state_change",    2, False),
]

LEVEL = 1

HEADER = """/* Auto-generated control event codes. DO NOT EDIT. */
#ifndef PT_EVENT
#define PT_EVENT
"""
FOOTER = "\n#endif /* PT_EVENT */\n"


def encode_event(num: int, nargs: int) -> str:
    code = (LEVEL << 28) | ((nargs & 0xF) << 24) | ((num & 0xFF) << 16)
    return f"0x{code:08X}"


def generate(out_path: str):
    lines = [HEADER]
    seen = set()

    for num, name, nargs, _ in EVENTS:
        if num in seen:
            raise SystemExit(f"错误：编号 0x{num:02X} 重复")
        seen.add(num)
        if not (0 <= num <= 255):
            raise SystemExit(f"错误：编号 {num} 超出 8-bit 范围")
        if not (0 <= nargs <= 15):
            raise SystemExit(f"错误：{name} 参数个数 {nargs} 超出 4-bit 范围")

        code = encode_event(num, nargs)
        macro = f"EVENT_{name.upper()}"
        lines.append(f"#define {macro:35s} {code}")

    lines.append(FOOTER)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"生成完成：{len(EVENTS)} 个事件 → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法：python3 gen_event.py <输出路径>")
    generate(sys.argv[1])
