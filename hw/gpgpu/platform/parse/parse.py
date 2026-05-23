"""
parse.py — 指令流二进制解析器
只认 level 字段，只写 level=0 的分支。

用法：python3 parse.py <输入二进制文件> <输出json文件>
"""

import sys
import struct
import json


# ============================================================
# 字段提取（通用）
# ============================================================

def extract_fields(inst_word, table):
    """根据字段表从 32-bit 指令字中提取字段。"""
    result = {}
    for name, start, width in table:
        mask = (1 << width) - 1
        result[name] = (inst_word >> start) & mask
    return result


# ============================================================
# 单条指令解析
# ============================================================

def parse_one(inst_word, operands):
    """解析单条指令，返回 dict。"""
    level = (inst_word >> 28) & 0xF

    if level == 0:
        table = [
            ("level",  28, 4),
            ("nargs",  24, 4),
            ("opcode", 16, 8),
            ("branch", 0,  1),
        ]
    else:
        table = [("level", 28, 4)]

    fields = extract_fields(inst_word, table)

    return {
        "fields":   fields,
        "operands": operands,
    }


# ============================================================
# 流解析
# ============================================================

def parse_stream(data):
    """
    从字节流中解析连续的指令记录。
    格式：inst_word(4B) + nargs×4B 操作数，重复。
    """
    records = []
    offset = 0

    while offset + 4 <= len(data):
        inst_word = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        nargs = (inst_word >> 24) & 0xF

        operands = []
        for _ in range(nargs):
            if offset + 4 > len(data):
                break
            operands.append(struct.unpack_from("<I", data, offset)[0])
            offset += 4

        records.append(parse_one(inst_word, operands))

    return records


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python3 pt_parse.py <输入文件> <输出文件>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "rb") as f:
        data = f.read()

    records = parse_stream(data)

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"解析完成：{len(records)} 条指令 → {out_path}")