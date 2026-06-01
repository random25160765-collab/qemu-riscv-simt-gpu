#!/usr/bin/env python3
"""fuzz-disasm: 给定 seed, dump 生成的指令序列 + 反汇编 + 分类"""
import sys, struct, subprocess, tempfile, os

SEED = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0x6a546a99

# ---- build the test binary once, then run ----
FUZZ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(FUZZ_DIR, 'fuzz_test')
if not os.path.exists(BIN):
    subprocess.check_call(['make', '-C', FUZZ_DIR, 'fuzz_test'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

p = subprocess.run([BIN, '-n', '1', '-s', str(SEED), '-d'],
                   capture_output=True, text=True, cwd=FUZZ_DIR)
out = p.stderr + p.stdout

# ---- extract instruction hex from output ----
import re
m = re.search(r'insts\(\d+\):(.*)', out)
if not m:
    print("No instructions found in output. Did the fuzzer crash immediately?")
    print("stderr:", out[:500])
    sys.exit(1)

hexes = m.group(1).strip().split()

# ---- disassemble ----
data = b''.join(struct.pack('<I', int(h, 16)) for h in hexes)
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(data); tmp = f.name

d = subprocess.check_output(
    ['riscv64-unknown-elf-objdump', '-b', 'binary', '-m', 'riscv:rv32', '-D', tmp],
    stderr=subprocess.DEVNULL).decode()
os.unlink(tmp)

asm_lines = {}
for l in d.strip().split('\n'):
    if ':' in l and '\t' in l:
        parts = l.strip().split('\t', 1)
        addr = int(parts[0].split(':')[0].strip(), 16)
        asm_lines[addr // 4] = parts[1].strip() if len(parts) > 1 else ''

# ---- classify ----
CAT = {
    0x33: 'ALU', 0x13: 'ALU-I', 0x37: 'LUI', 0x17: 'AUIPC',
    0x03: 'LOAD', 0x23: 'STORE', 0x63: 'BR', 0x6F: 'JAL', 0x67: 'JALR',
    0x43: 'FMA', 0x47: 'FMA', 0x4B: 'FMA', 0x4F: 'FMA',
    0x53: 'FP', 0x73: 'CSR', 0x07: 'FLW', 0x27: 'FSW',
    0x0B: 'CUSTOM!', 0x2B: 'CUSTOM!', 0x5B: 'CUSTOM!', 0x7B: 'CUSTOM!',
}

print(f"Seed: 0x{SEED:08x}  ({SEED})")
print(f"Instructions: {len(hexes)}")
print()
for i, h in enumerate(hexes):
    v = int(h, 16)
    op = v & 0x7F
    cat = CAT.get(op, 'UNK')
    asm = asm_lines.get(i, '???')
    rd = (v >> 7) & 0x1F
    rs1 = (v >> 15) & 0x1F
    rs2 = (v >> 20) & 0x1F
    marker = ''
    if op in (0x0B, 0x2B, 0x5B, 0x7B): marker = ' ⚠ CUSTOM OPCODE'
    if asm.startswith('.insn') or asm == '???': marker = ' ⚠ INVALID'
    if rd == 0 and cat in ('ALU','ALU-I','FP','CSR'): marker += ' [writes x0]'
    print(f"  [{i:3d}] 0x{h}  {cat:8s} {asm:45s} rd={rd:2d} rs1={rs1:2d} rs2={rs2:2d}{marker}")

# ---- summary ----
bad = [h for h in hexes if (int(h,16) & 0x7F) in (0x0B, 0x2B, 0x5B, 0x7B)]
inv = [h for h in hexes if asm_lines.get(hexes.index(h), '').startswith('.insn')]
print(f"\nSummary: {len(bad)} custom-opcode, {len(inv)} invalid-encoding")
if p.returncode != 0:
    print(f"STATUS: CRASH (exit={p.returncode})")
else:
    print("STATUS: OK")
