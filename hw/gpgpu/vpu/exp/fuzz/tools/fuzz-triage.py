#!/usr/bin/env python3
"""fuzz-triage: crash seed 一键分析, 定位崩溃指令 (via gdb)"""
import sys, os, subprocess, re, struct, tempfile

SEED = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0x6a546a99
FUZZ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(FUZZ_DIR, 'fuzz_test')

print(f"=== Triage seed=0x{SEED:08x} ===")

# 1. build if needed
if not os.path.exists(BIN):
    subprocess.check_call(['make', '-C', FUZZ_DIR, 'fuzz_test'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 2. get instruction sequence
p = subprocess.run([BIN, '-n', '1', '-s', str(SEED), '-d'],
                   capture_output=True, text=True, cwd=FUZZ_DIR)
out = p.stderr + p.stdout
m = re.search(r'insts\(\d+\):(.*)', out)
if not m:
    print("ERROR: no instruction dump. Does fuzzer crash before engine_exec?")
    print(out[-500:])
    sys.exit(1)

hexes = m.group(1).strip().split()
print(f"Instructions: {len(hexes)}")

# 3. disassemble
data = b''.join(struct.pack('<I', int(h, 16)) for h in hexes)
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(data); tmp = f.name
d = subprocess.check_output(
    ['riscv64-unknown-elf-objdump', '-b', 'binary', '-m', 'riscv:rv32', '-D', tmp],
    stderr=subprocess.DEVNULL).decode()
os.unlink(tmp)

asm = {}
for l in d.strip().split('\n'):
    if ':' in l and '\t' in l:
        addr = int(l.split(':')[0].strip(), 16) // 4
        asm[addr] = l.split('\t')[1].strip() if '\t' in l else ''

# 4. gdb crash location
gdb_out = subprocess.run(
    ['gdb', '-batch', '-ex', 'run', '-ex', 'bt', '-ex', 'info line',
     '--args', BIN, '-n', '1', '-s', str(SEED)],
    capture_output=True, text=True, cwd=FUZZ_DIR, timeout=10).stderr + ''
# try stdout too
gdb_out2 = subprocess.run(
    ['gdb', '-batch', '-ex', 'run', '-ex', 'bt',
     '--args', BIN, '-n', '1', '-s', str(SEED)],
    capture_output=True, text=True, cwd=FUZZ_DIR, timeout=10).stdout

gdb_all = gdb_out + gdb_out2
# find crash PC
pc_m = re.search(r'(?:0x[0-9a-f]+ in|at) .*?engine\.c:(\d+)', gdb_all)
crash_line = int(pc_m.group(1)) if pc_m else None

# find which instruction was at the crash
# the crash is inside engine_exec, we need to map ThOp index
# heuristic: check if ebreak was reached
ebreak_idx = None
for i, h in enumerate(hexes):
    if int(h, 16) == 0x00100073:
        ebreak_idx = i
        break

print(f"\nCrash analysis:")
print(f"  Exit code: {p.returncode}")
if crash_line:
    print(f"  Crash line: engine.c:{crash_line}")
print(f"  ebreak at index: {ebreak_idx}")

# show suspicious instructions
print(f"\nSuspicious instructions:")
for i, h in enumerate(hexes):
    v = int(h, 16)
    op = v & 0x7F
    a = asm.get(i, '???')
    flags = []
    if op in (0x0B, 0x2B, 0x5B, 0x7B): flags.append('CUSTOM-OPCODE')
    if a.startswith('.insn'): flags.append('INVALID-ENC')
    if (v >> 7) & 0x1F == 0 and op not in (0x73, 0x63): flags.append('WRITES-X0')
    if flags:
        print(f"  [{i:3d}] 0x{h}  {a[:50]}")
        for f in flags: print(f"         ⚠ {f}")
        print()

# verdict
if any((int(h,16) & 0x7F) in (0x0B,0x2B,0x5B,0x7B) for h in hexes):
    print("VERDICT: Custom opcode instructions in sequence → add to badlist")
elif crash_line and crash_line > 650:
    print(f"VERDICT: Crash in fused/texture handler at engine.c:{crash_line}")
else:
    print(f"VERDICT: Unknown crash cause, investigate engine.c:{crash_line}")
