#!/usr/bin/env python3
"""fuzz-bisect: 二分定位崩溃指令。使用 fuzz_test -r 快速测试子序列。"""
import sys, os, subprocess, re, struct, tempfile

FUZZ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(FUZZ_DIR, 'fuzz_test')
CRASH_DIR = os.path.join(FUZZ_DIR, 'crashes')

def load_raw(seed_str):
    fname = os.path.join(CRASH_DIR, f'seed_{seed_str}.txt')
    if os.path.exists(fname):
        with open(fname) as f:
            m = re.search(r'insts\(\d+\):(.*)', f.read())
            if m: return [int(x,16) for x in m.group(1).strip().split()]
    p = subprocess.run([BIN, '-n','1','-s',str(int(seed_str,0)),'-d'],
                       capture_output=True, text=True, cwd=FUZZ_DIR, timeout=10)
    m = re.search(r'insts\(\d+\):(.*)', p.stderr + p.stdout)
    if m: return [int(x,16) for x in m.group(1).strip().split()]
    return None

def test(raw_list):
    """调用 fuzz_test -r, 返回 (crashed, exit_code)"""
    hexes = ' '.join(f'0x{x:08x}' for x in raw_list)
    r = subprocess.run([BIN, '-r', hexes, '-q'], capture_output=True,
                       timeout=5, cwd=FUZZ_DIR)
    crashed = r.returncode != 0 and r.returncode != 1
    return crashed, r.returncode

def disasm(inst):
    data = struct.pack('<I', inst)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(data); tmp = f.name
    asm = subprocess.check_output(
        ['riscv64-unknown-elf-objdump','-b','binary','-m','riscv:rv32','-D',tmp],
        stderr=subprocess.DEVNULL).decode()
    os.unlink(tmp)
    for l in asm.strip().split('\n')[1:]:
        if ':' in l: return l.split('\t')[1].strip() if '\t' in l else l.strip()
    return '???'

def bisect(raw, depth=0):
    pref = "  " * depth
    n = len(raw)
    if n <= 1: return None

    crashed, _ = test(raw)
    if not crashed:
        print(f"{pref}[{n}] OK → sequence fixed by removal?")
        return None

    # 去后半
    mid = n // 2
    fh = raw[:mid] + [0x00100073]
    if test(fh)[0]:
        print(f"{pref}[{n}] first half [{mid}] crashes")
        return bisect(raw[:mid], depth+1)

    # 去前半
    sh = raw[mid:] if raw[mid:][-1] == 0x00100073 else raw[mid:] + [0x00100073]
    if test(sh)[0]:
        print(f"{pref}[{n}] second half [{n-mid}] crashes")
        return bisect(raw[mid:], depth+1)

    # 逐条测试
    for i in range(len(raw)-1):
        s = raw[:i] + raw[i+1:]
        if not test(s)[0]:
            print(f"{pref}[{n}] remove inst[{i}] 0x{raw[i]:08x} fixes crash")
            return raw[i]

    print(f"{pref}[{n}] interaction: cannot reduce to single instruction")
    return raw

if __name__ == '__main__':
    s = sys.argv[1] if len(sys.argv) > 1 else '0x1'
    raw = load_raw(s)
    if not raw: print("ERROR: no sequence"); sys.exit(1)
    if raw[-1] == 0x00100073: raw = raw[:-1]
    print(f"=== Bisect seed={s}  {len(raw)} insts ===\n")
    r = bisect(raw)
    print()
    if r is None: print("RESULT: no crash (fixed?)")
    elif isinstance(r, int):
        print(f"CRASHING INSTRUCTION: 0x{r:08x}  {disasm(r)}")
    else:
        print(f"MINIMAL SET: {len(r)} insts")
        for x in r: print(f"  0x{x:08x}  {disasm(x)}")
