#!/usr/bin/env python3
"""
customasm.py — RISC-V Custom Instruction Assembler

Preprocesses .S files: replaces VPU/TCU/SFU/LP mnemonics with .word directives.
Shares encoding definitions with engine via dispatch.h INSTRUCTION_LIST.

Usage:
  python3 tools/customasm.py kernels/vecmul_vpu.S > kernels/vecmul_vpu.asm.S
"""

import re, sys, os

# ── Register maps ──────────────────────────────────────────────
GPR = { 'zero':0,'ra':1,'sp':2,'gp':3,'tp':4,'t0':5,'t1':6,'t2':7,
        's0':8,'s1':9,'a0':10,'a1':11,'a2':12,'a3':13,'a4':14,'a5':15,
        'a6':16,'a7':17,'s2':18,'s3':19,'s4':20,'s5':21,'s6':22,'s7':23,
        's8':24,'s9':25,'s10':26,'s11':27,'t3':28,'t4':29,'t5':30,'t6':31 }
for i in range(32): GPR[f'x{i}'] = i

FPR = {}
for i in range(32): FPR[f'f{i}'] = i
# Custom VPU: v0-v15 → FPR 16-31 (parse_reg uses this)
for i in range(16): FPR[f'v{i}'] = 16 + i

# Standard RVV: v0-v31 → VPR 0-31 (separate register file, used directly by RVV paths)
RVV_VREG = {}
for i in range(32): RVV_VREG[f'v{i}'] = i

ACC = {}
for i in range(8): ACC[f'acc{i}'] = i

def parse_reg(s):
    s = s.strip()
    if s in GPR: return (False, GPR[s])
    if s in FPR: return (True, FPR[s])
    if s in ACC: return (True, ACC[s])
    return (False, None)

def strip_paren(s):
    return s.strip().lstrip('(').rstrip(')')

# ── Encoding table from dispatch.h ──────────────────────────────

def load_from_dispatch(path):
    """Parse INSTRUCTION_LIST from dispatch.h → {name: (opcode,funct3,funct7,fmt)}"""
    with open(path) as f:
        content = f.read()

    m = re.search(r'#define\s+INSTRUCTION_LIST\s+(.*?)(?=\n\n|\nstatic|\n/\*|\Z)', content, re.DOTALL)
    if not m: return {}

    entries = {}
    for match in re.finditer(r'X\((\w+),\s*"([^"]+)"', m.group(1)):
        name = match.group(1)
        pat  = match.group(2)
        bits = pat.replace(' ', '')
        if len(bits) != 32: continue

        # Extract fields (pattern is MSB-first: funct7[6:0], rs2[4:0], rs1[4:0], funct3[2:0], rd[4:0], opcode[6:0])
        f7_str  = bits[0:7]
        rs2_str = bits[7:12]
        rs1_str = bits[12:17]
        f3_str  = bits[17:20]
        rd_str  = bits[20:25]
        op_str  = bits[25:32]

        def fixed(s):
            return all(c in '01' for c in s)

        if not fixed(op_str): continue  # need fixed opcode

        opcode = int(op_str, 2)
        funct7 = int(f7_str, 2) if fixed(f7_str) else 0
        funct3 = int(f3_str, 2) if fixed(f3_str) else 0  # default '?'→0

        # Determine format from variable fields
        rd_var  = not fixed(rd_str)
        rs1_var = not fixed(rs1_str)
        rs2_var = not fixed(rs2_str)

        fixed_rs2 = int(rs2_str, 2) if fixed(rs2_str) else 0

        if not rd_var and not rs1_var and not rs2_var:
            fmt = 'N'   # no operands (ebreak, barrier)
        elif rd_var and rs1_var and rs2_var:
            fmt = 'R'   # 3 registers (vfadd.v vd, vs1, vs2)
        elif rd_var and rs1_var and not rs2_var:
            fmt = 'R2'  # 2 registers (vfexp.v vd, vs1)
        elif rd_var and not rs1_var and not rs2_var:
            fmt = 'R1'  # 1 register (mma.zero acc)
        else:
            fmt = 'R'

        entries[name] = (opcode, funct3, funct7, fmt, fixed_rs2)

    return entries

# Load from dispatch.h (single source of truth)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DISPATCH_H = os.path.join(SCRIPT_DIR, '..', 'core', 'dispatch.h')
ALL_TABLE = load_from_dispatch(DISPATCH_H)

# Only convert custom/SFU/LP instructions, NOT standard RISC-V
CUSTOM_OPCODES = {0x0B, 0x2B}  # custom-0, custom-1
SFU_LP_FUNCT7  = {0b0110000, 0b0100010, 0b0100100, 0b0100110}  # SFU, bf16, e4m3/e5m2, e2m1

TABLE = {}
for name, (opcode, funct3, funct7, fmt, fixed_rs2) in ALL_TABLE.items():
    if opcode in CUSTOM_OPCODES:
        TABLE[name] = (opcode, funct3, funct7, fmt, fixed_rs2)
    elif opcode == 0x53 and funct7 in SFU_LP_FUNCT7:
        TABLE[name] = (opcode, funct3, funct7, fmt, fixed_rs2)

# Map assembly mnemonics: _ → . (e.g. mma_zero → mma.zero)
MNEMONIC_MAP = {}
for name, info in TABLE.items():
    MNEMONIC_MAP[name] = info
    dotted = name.replace('_', '.')
    if dotted != name:
        MNEMONIC_MAP[dotted] = info
    if name.endswith('_s'):
        MNEMONIC_MAP[name.replace('_s', '.s')] = info
    if name.endswith('_vs'):
        MNEMONIC_MAP[name.replace('_vs', '.vs')] = info
    elif name.endswith('_v') and not name.endswith('.v'):
        MNEMONIC_MAP[name.replace('_v', '.v')] = info

# ── Standard RVV OPIVV (opcode 0x57, funct3=000) ────────────────
RVV_OPIVV = {
    'vadd.vv':   0b000000,
    'vsub.vv':   0b000010,
    'vmin.vv':   0b000101,
    'vminu.vv':  0b000100,
    'vmax.vv':   0b000111,
    'vmaxu.vv':  0b000110,
    'vand.vv':   0b001001,
    'vor.vv':    0b001010,
    'vxor.vv':   0b001011,
    'vsll.vv':   0b100101,
    'vsrl.vv':   0b101000,
    'vsra.vv':   0b101001,
    'vmseq.vv':  0b011000,
    'vmsne.vv':  0b011001,
    'vmsltu.vv': 0b011010,
    'vmslt.vv':  0b011011,
    'vmerge.vvm':0b010111,
}
for mnem, f6 in RVV_OPIVV.items():
    MNEMONIC_MAP[mnem] = ('RVV', f6, 1)  # ('RVV', funct6, vm)
# vmerge.vvm requires vm=0
MNEMONIC_MAP['vmerge.vvm'] = ('RVV', 0b010111, 0)

# RVV unit-stride load/store (funct3=001/010, funct6=0, vm=1)
MNEMONIC_MAP['vle32.v'] = ('RVVL', 0b000000, 1)   # ('RVVL', funct6, vm)
MNEMONIC_MAP['vse32.v'] = ('RVVS', 0b000000, 1)   # ('RVVS', funct6, vm)
MNEMONIC_MAP['vsetvli'] = ('VSET', 0, 0)           # rd(GPR), rs1(GPR), special encoding

# RVV OPFVV (funct3=001) and OPFVF (funct3=101)
MNEMONIC_MAP['vfmul.vv'] = ('RVV', 0b100100, 1)   # vd, vs1, vs2
MNEMONIC_MAP['vfmul.vf'] = ('RVVF', 0b100100, 1)  # vd, vs2(vec), rs1(scalar) — special

# ── Assembler ───────────────────────────────────────────────────

def encode_r(opcode, funct3, funct7, rd, rs1, rs2):
    return (funct7 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 7) << 12) | ((rd & 0x1F) << 7) | opcode

def encode_rvv(funct6, vm, vd, vs1, vs2):
    return (funct6 << 26) | ((vm & 1) << 25) | ((vs2 & 0x1F) << 20) | ((vs1 & 0x1F) << 15) | (0 << 12) | ((vd & 0x1F) << 7) | 0x57

def assemble(name, operands):
    """Assemble one instruction → 32-bit word (or None if not custom/RVV)."""
    if name not in MNEMONIC_MAP:
        return None
    info = MNEMONIC_MAP[name]

    # Standard RVV path: vadd.vv vd, vs1, vs2 etc.
    if info[0] == 'RVV':
        _, funct6, vm = info
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 3: return None
        vd  = RVV_VREG.get(cleaned[0])
        vs1 = RVV_VREG.get(cleaned[1])
        vs2 = RVV_VREG.get(cleaned[2])
        if vd is None or vs1 is None or vs2 is None: return None
        return encode_rvv(funct6, vm, vd, vs1, vs2)

    # RVV load: vle32.v vd, (rs1) → vd=VPR, rs1=GPR, opcode=0x07 funct3=6
    if info[0] == 'RVVL':
        _, funct6, vm = info
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 2: return None
        vd  = RVV_VREG.get(cleaned[0])
        rs1 = GPR.get(cleaned[1])
        if vd is None or rs1 is None: return None
        return (funct6 << 26) | ((vm & 1) << 25) | ((rs1 & 0x1F) << 15) | (6 << 12) | ((vd & 0x1F) << 7) | 0x07

    # RVV store: vse32.v vs3, (rs1) → vs3=VPR, rs1=GPR, opcode=0x27 funct3=6
    if info[0] == 'RVVS':
        _, funct6, vm = info
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 2: return None
        vs3 = RVV_VREG.get(cleaned[0])
        rs1 = GPR.get(cleaned[1])
        if vs3 is None or rs1 is None: return None
        return (funct6 << 26) | ((vm & 1) << 25) | ((rs1 & 0x1F) << 15) | (6 << 12) | ((vs3 & 0x1F) << 7) | 0x27

    # RVV OPFVF: vfmul.vf vd, vs2, rs1 → vd=VPR, vs2=VPR, rs1=scalar FPR
    if info[0] == 'RVVF':
        _, funct6, vm = info
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 3: return None
        vd  = RVV_VREG.get(cleaned[0])
        vs2 = RVV_VREG.get(cleaned[1])
        rs1 = FPR.get(cleaned[2]) if cleaned[2] in FPR else GPR.get(cleaned[2])
        if vd is None or vs2 is None or rs1 is None: return None
        return (funct6 << 26) | ((vm & 1) << 25) | ((vs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (5 << 12) | ((vd & 0x1F) << 7) | 0x57

    # vsetvli: vsetvli rd, rs1 → rd=GPR, rs1=GPR, zimm=0, funct3=7, opcode=0x57
    if info[0] == 'VSET':
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 2: return None
        rd  = GPR.get(cleaned[0])
        rs1 = GPR.get(cleaned[1])
        if rd is None or rs1 is None: return None
        return ((rs1 & 0x1F) << 15) | (7 << 12) | ((rd & 0x1F) << 7) | 0x57

    # RVV unary: vfsig.v vd, vs1 → vd=VPR, vs1=VPR
    if info[0] == 'RVVU':
        _, funct6, vm = info
        cleaned = [strip_paren(op) for op in operands if strip_paren(op) and strip_paren(op) != ',']
        if len(cleaned) < 2: return None
        vd  = RVV_VREG.get(cleaned[0])
        vs1 = RVV_VREG.get(cleaned[1])
        if vd is None or vs1 is None: return None
        return (funct6 << 26) | ((vm & 1) << 25) | ((vs1 & 0x1F) << 15) | (1 << 12) | ((vd & 0x1F) << 7) | 0x57

    # Custom path
    opcode, funct3, funct7, fmt, fixed_rs2 = info
    opcode, funct3, funct7, fmt, fixed_rs2 = MNEMONIC_MAP[name]

    # Clean operands: strip parens, skip commas
    cleaned = []
    for op in operands:
        op = strip_paren(op)
        if op and op != ',':
            cleaned.append(op)

    if fmt == 'N':
        return encode_r(opcode, funct3, funct7, 0, 0, 0)

    if fmt in ('R', 'R1', 'R2'):
        if fmt == 'R' and len(cleaned) < 3:
            return None
        if fmt == 'R1' and len(cleaned) < 1:
            return None
        if fmt == 'R2' and len(cleaned) < 2:
            return None

        rd = parse_reg(cleaned[0])[1]
        rs1 = parse_reg(cleaned[1])[1] if (fmt in ('R', 'R2') and len(cleaned) > 1) else 0
        rs2 = parse_reg(cleaned[2])[1] if (fmt == 'R' and len(cleaned) > 2) else fixed_rs2

        if rd is None or (fmt in ('R', 'R2') and rs1 is None):
            return None
        if fmt == 'R' and rs2 is None:
            return None

        return encode_r(opcode, funct3, funct7, rd, rs1, rs2)

    return None

# ── Preprocessor ────────────────────────────────────────────────

def preprocess(text):
    """Replace custom mnemonics with .word directives."""
    out = []
    for line in text.split('\n'):
        stripped = line.strip()

        if (not stripped or stripped.startswith('.') or stripped.startswith('#')
            or stripped.startswith('//') or stripped.endswith(':')):
            out.append(line)
            continue

        # Remove inline comment
        if '#' in stripped:
            comment_start = stripped.index('#')
            parts_str = stripped[:comment_start]
            comment = stripped[comment_start:]
        else:
            parts_str = stripped
            comment = ''

        # Tokenize: split by commas and spaces
        tokens = parts_str.replace(',', ' ').split()
        if len(tokens) < 1:
            out.append(line)
            continue

        mnemonic = tokens[0]
        operands = tokens[1:]

        word = assemble(mnemonic, operands)
        if word is not None:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f'{indent}.word 0x{word:08X}{comment}')
        else:
            out.append(line)

    return '\n'.join(out)

# ── CLI ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.S> [output.S]", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        text = f.read()

    result = preprocess(text)

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            f.write(result)
        print(f"Written: {sys.argv[2]}", file=sys.stderr)
    else:
        print(result)
