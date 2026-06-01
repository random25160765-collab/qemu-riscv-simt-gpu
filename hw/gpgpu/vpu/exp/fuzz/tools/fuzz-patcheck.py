#!/usr/bin/env python3
"""fuzz-patcheck: 验证所有 safe_ops pattern, 检查 bit 位宽, 空 pattern, opcode 分布"""
import sys, os, subprocess, re, struct, tempfile

FUZZ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 1. 从 fuzz_test 的 constructor 输出获取 n_safe ----
# 直接解析 dispatch.h 的 INSTRUCTION_LIST, 模拟 fuzzer 的 _init 过滤逻辑

dispatch_h = os.path.join(FUZZ_DIR, '..', 'core', 'dispatch.h')
with open(dispatch_h) as f:
    content = f.read()

# 提取 INSTRUCTION_LIST
m = re.search(r'#define INSTRUCTION_LIST \\\n(.*?)(?=\n\S|\Z)', content, re.DOTALL)
if not m:
    print("ERROR: cannot find INSTRUCTION_LIST")
    sys.exit(1)
body = m.group(1)

# 解析每行 X(name, "pattern", TYPE, immFn);
entries = []
for line in body.split('\n'):
    line = line.strip()
    if not line.startswith('X('):
        continue
    # 提取 name 和 pattern
    name_m = re.match(r'X\((\w+),\s*"([^"]+)"', line)
    if name_m:
        entries.append((name_m.group(1), name_m.group(2)))

# fuzzer 的黑名单
BAD = {"jal","jalr","beq","bne","blt","bge","bltu","bgeu",
       "lb","lh","lw","lbu","lhu","sb","sh","sw","flw","fsw",
       "csrrw","csrrs","csrrc","csrrwi","csrrsi","csrrci","ebreak",
       "lr_w","sc_w","amoswap_w","amoadd_w","amoxor_w",
       "amoand_w","amoor_w","amomin_w","amomax_w","amominu_w","amomaxu_w",
       "fused_ld2_fma","fused_scal_mul","fused_gelu","fused_softmax","fused_matmul_loop"}

safe = [(n, p) for n, p in entries if n not in BAD]
print(f"Total instructions: {len(entries)}")
print(f"Blocked: {len(entries) - len(safe)}")
print(f"Safe ops: {len(safe)}")
print()

# ---- 2. 检查每个 pattern ----
issues = []
opcode_dist = {}
for name, pat in safe:
    # 空 pattern
    if not pat or pat.strip() == '':
        issues.append(f"EMPTY: {name}")
        continue

    # bit 计数
    non_space = [c for c in pat if c != ' ']
    nbits = len(non_space)
    if nbits != 32:
        issues.append(f"BITCOUNT={nbits}: {name}  '{pat}'")

    # 检查 pattern 的 opcode (最后 7 个非空格字符)
    if nbits == 32:
        op_bits = ''.join(non_space[-7:])
        if '?' in op_bits:
            issues.append(f"OP_HAS_?: {name}  opcode_bits={op_bits}")

    # opcode 分布
    if nbits == 32:
        # 模拟: 如果所有 ? 是 0, opcode 是多少
        test_pat = ''.join(c if c != '?' else '0' for c in pat)
        v = 0; b = 31
        for c in test_pat:
            if c == ' ': continue
            if c == '1': v |= 1 << b
            b -= 1
        op = v & 0x7F
        opcode_dist[op] = opcode_dist.get(op, 0) + 1

# ---- 3. 打印 ----
for iss in issues:
    print(f"  ISSUE: {iss}")

print(f"\nOpcode distribution (min encoding):")
for op in sorted(opcode_dist):
    print(f"  0x{op:02x}: {opcode_dist[op]} instructions")

print(f"\nTotal issues: {len(issues)}")
sys.exit(1 if issues else 0)
