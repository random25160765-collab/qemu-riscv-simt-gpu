#!/usr/bin/env python3
"""
gen_rvv.py — RVV instruction code generator (SEW-aware)

Each instruction produces:
  op_<name>:    dispatch stub → goto correct SEW variant
  op_<name>_e8/e16/e32:  compiled with SEW as constant

SEW=32 handler uses existing FOR_EACH_LANE pattern (backward compat path).
SEW=8/16 handlers add inner per-element loop RVV_EACH.
"""
import os, sys

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "core")
OUT_DISPATCH = os.path.join(OUT_DIR, "rvv_dispatch.h")
OUT_HANDLERS = os.path.join(OUT_DIR, "rvv_handlers.h")
SCRIPT = os.path.basename(__file__)

# ═══════════════════════════════════════════════════════════════
#  INSTRUCTION TABLE
#  cat:  OPIVV | OPIVX | OPFVV | OPFVF | OPMVV
#  sew:  bitmask of supported SEW
#  body: C expression. Variables: a(vs1)=T, b(vs2)=T, sc=T(scalar)
#        Use (T) cast explicitly for signed, (U) for unsigned.
#        For comparison→mask: use T types, result -1 or 0
# ═══════════════════════════════════════════════════════════════

INSTRUCTIONS = [
    # ── OPIVV ──────────────────────────────────────────────────
    ("vadd.vv",  0b000000, "OPIVV", 8|16|32, "((T)(a) + (T)(b))"),
    ("vsub.vv",  0b000010, "OPIVV", 8|16|32, "((T)(a) - (T)(b))"),
    ("vand.vv",  0b001001, "OPIVV", 8|16|32, "((T)(a) & (T)(b))"),
    ("vor.vv",   0b001010, "OPIVV", 8|16|32, "((T)(a) | (T)(b))"),
    ("vxor.vv",  0b001011, "OPIVV", 8|16|32, "((T)(a) ^ (T)(b))"),
    ("vsll.vv",  0b100101, "OPIVV", 8|16|32, "((T)(a) << ((U)(b) & (SEW-1)))"),
    ("vsrl.vv",  0b101000, "OPIVV", 8|16|32, "((U)(a) >> ((U)(b) & (SEW-1)))"),
    ("vsra.vv",  0b101001, "OPIVV", 8|16|32, "((T)(a) >> ((U)(b) & (SEW-1)))"),
    ("vmseq.vv", 0b011000, "OPIVV", 8|16|32, "(((T)(a) == (T)(b)) ? ~(T)(0) : (T)(0))"),
    ("vmsne.vv", 0b011001, "OPIVV", 8|16|32, "(((T)(a) != (T)(b)) ? ~(T)(0) : (T)(0))"),
    ("vmslt.vv", 0b011011, "OPIVV", 8|16|32, "(((T)(a) < (T)(b)) ? ~(T)(0) : (T)(0))"),
    ("vmsltu.vv",0b011010, "OPIVV", 8|16|32, "(((U)(a) < (U)(b)) ? ~(T)(0) : (T)(0))"),
    ("vmin.vv",  0b000101, "OPIVV", 8|16|32, "(((T)(a) < (T)(b)) ? (T)(a) : (T)(b))"),
    ("vminu.vv", 0b000100, "OPIVV", 8|16|32, "(((U)(a) < (U)(b)) ? (U)(a) : (U)(b))"),
    ("vmax.vv",  0b000111, "OPIVV", 8|16|32, "(((T)(a) > (T)(b)) ? (T)(a) : (T)(b))"),
    ("vmaxu.vv", 0b000110, "OPIVV", 8|16|32, "(((U)(a) > (U)(b)) ? (U)(a) : (U)(b))"),
    # ── OPIVX (vs2=vector, rs1=GPR scalar) ─────────────────────
    ("vadd.vx",  0b000000, "OPIVX", 8|16|32, "((T)(b) + sc)"),
    ("vsub.vx",  0b000010, "OPIVX", 8|16|32, "((T)(b) - sc)"),
    ("vand.vx",  0b001001, "OPIVX", 8|16|32, "((T)(b) & sc)"),
    ("vor.vx",   0b001010, "OPIVX", 8|16|32, "((T)(b) | sc)"),
    ("vxor.vx",  0b001011, "OPIVX", 8|16|32, "((T)(b) ^ sc)"),
    ("vsll.vx",  0b100101, "OPIVX", 8|16|32, "((T)(b) << ((U)(sc) & (SEW-1)))"),
    ("vsrl.vx",  0b101000, "OPIVX", 8|16|32, "((U)(b) >> ((U)(sc) & (SEW-1)))"),
    ("vsra.vx",  0b101001, "OPIVX", 8|16|32, "((T)(b) >> ((U)(sc) & (SEW-1)))"),
    ("vmseq.vx", 0b011000, "OPIVX", 8|16|32, "(((T)(b) == sc) ? ~(T)(0) : (T)(0))"),
    ("vmsne.vx", 0b011001, "OPIVX", 8|16|32, "(((T)(b) != sc) ? ~(T)(0) : (T)(0))"),
    ("vmslt.vx", 0b011011, "OPIVX", 8|16|32, "(((T)(b) < sc) ? ~(T)(0) : (T)(0))"),
    ("vmsltu.vx",0b011010, "OPIVX", 8|16|32, "(((U)(b) < (U)(sc)) ? ~(T)(0) : (T)(0))"),
    # ── OPIVI integer immediate (sc=5-bit sign-extended imm from bits[19:15]) ──
    ("vadd.vi",   0b000000, "OPIVI", 8|16|32, "((T)(b) + (T)(sc))"),
    ("vrsub.vi",  0b000011, "OPIVI", 8|16|32, "((T)(sc) - (T)(b))"),
    ("vand.vi",   0b011001, "OPIVI", 8|16|32, "((T)(b) & (T)(sc))"),
    ("vor.vi",    0b011010, "OPIVI", 8|16|32, "((T)(b) | (T)(sc))"),
    ("vxor.vi",   0b011011, "OPIVI", 8|16|32, "((T)(b) ^ (T)(sc))"),
    ("vsll.vi",   0b100101, "OPIVI", 8|16|32, "((T)(b) << ((U)(sc) & (SEW-1)))"),
    ("vsrl.vi",   0b101000, "OPIVI", 8|16|32, "((U)(b) >> ((U)(sc) & (SEW-1)))"),
    ("vsra.vi",   0b101001, "OPIVI", 8|16|32, "((T)(b) >> ((U)(sc) & (SEW-1)))"),
    ("vmseq.vi",  0b011000, "OPIVI", 8|16|32, "(((T)(b) == (T)(sc)) ? ~(T)(0) : (T)(0))"),
    ("vmsne.vi",  0b011001, "OPIVI", 8|16|32, "(((T)(b) != (T)(sc)) ? ~(T)(0) : (T)(0))"),
    # ── OPIVV saturating ───────────────────────────────────────
    ("vsadd.vv",  0b100001, "OPIVV", 8|16|32, "RV_SAT_ADD_S(a, b)"),
    ("vsaddu.vv", 0b100000, "OPIVV", 8|16|32, "RV_SAT_ADD_U(a, b)"),
    ("vssub.vv",  0b100011, "OPIVV", 8|16|32, "RV_SAT_SUB_S(a, b)"),
    ("vssubu.vv", 0b100010, "OPIVV", 8|16|32, "RV_SAT_SUB_U(a, b)"),
    # ── OPIVV widening (inputs SEW, output 2*SEW) ──────────────
    ("vwadd.vv",   0b110000, "OPIVV", 8|16,   "((W)(T)(a) + (W)(T)(b))"),
    ("vwaddu.vv",  0b110001, "OPIVV", 8|16,   "((W)(U)(a) + (W)(U)(b))"),
    ("vwsub.vv",   0b110010, "OPIVV", 8|16,   "((W)(T)(a) - (W)(T)(b))"),
    ("vwsubu.vv",  0b110011, "OPIVV", 8|16,   "((W)(U)(a) - (W)(U)(b))"),
    ("vwmul.vv",   0b111010, "OPIVV", 8|16,   "((W)(T)(a) * (W)(T)(b))"),
    ("vwmulu.vv",  0b111000, "OPIVV", 8|16,   "((W)(U)(a) * (W)(U)(b))"),
    ("vwmulsu.vv", 0b111011, "OPIVV", 8|16,   "((int64_t)(T)(a) * (int64_t)(U)(b))"),
    # ── OPIVV narrowing (input 2*SEW, output SEW) ──────────────
    ("vnsrl.wv",   0b101100, "OPIVV", 8|16,   "((T)((U)(a) >> ((U)(b) & (2*SEW-1))))"),
    ("vnsra.wv",   0b101101, "OPIVV", 8|16,   "((T)((T)(a) >> ((U)(b) & (2*SEW-1))))"),
    # ── OPMVV integer multiply ─────────────────────────────────
    ("vmul.vv",    0b100101, "OPMVV", 8|16|32, "((T)(a) * (T)(b))"),
    ("vmulh.vv",   0b100111, "OPMVV", 8|16|32, "((T)((int64_t)(T)(a) * (int64_t)(T)(b) >> SEW))"),
    ("vmulhu.vv",  0b101001, "OPMVV", 8|16|32, "((T)((uint64_t)(U)(a) * (uint64_t)(U)(b) >> SEW))"),
    ("vmulhsu.vv", 0b101010, "OPMVV", 8|16|32, "((T)((int64_t)(T)(a) * (uint64_t)(U)(b) >> SEW))"),
    ("vmacc.vv",   0b101101, "OPMVV", 8|16|32, "(c + (T)(a) * (T)(b))"),
    ("vnmsac.vv",  0b101111, "OPMVV", 8|16|32, "(c - (T)(a) * (T)(b))"),
    # ── OPIVV integer division ─────────────────────────────────
    ("vdiv.vv",    0b100000, "OPIVV", 8|16|32, "((T)(b) ? (T)(a) / (T)(b) : (T)-1)"),
    ("vdivu.vv",   0b100001, "OPIVV", 8|16|32, "((U)(b) ? (U)(a) / (U)(b) : (U)-1)"),
    ("vrem.vv",    0b100010, "OPIVV", 8|16|32, "((T)(b) ? (T)(a) % (T)(b) : (T)(a))"),
    ("vremu.vv",   0b100011, "OPIVV", 8|16|32, "((U)(b) ? (U)(a) % (U)(b) : (U)(a))"),
    # ── OPFVV float-vector (SEW=32 only, T=float, U=uint32_t) ──
    ("vfadd.vv",   0b000000, "OPFVV", 32, "(a + b)"),
    ("vfsub.vv",   0b000010, "OPFVV", 32, "(a - b)"),
    ("vfmul.vv",   0b100100, "OPFVV", 32, "(a * b)"),
    ("vfdiv.vv",   0b100000, "OPFVV", 32, "(a / b)"),
    ("vfmin.vv",   0b000100, "OPFVV", 32, "((a < b) ? a : b)"),
    ("vfmax.vv",   0b000110, "OPFVV", 32, "((a > b) ? a : b)"),
    ("vfsgnj.vv",  0b001000, "OPFVV", 32, "({ uint32_t _a=*(uint32_t*)&a, _b=*(uint32_t*)&b; uint32_t _r = (_a & 0x7FFFFFFF) | (_b & 0x80000000); *(float*)&_r; })"),
    ("vfsgnjn.vv", 0b001001, "OPFVV", 32, "({ uint32_t _a=*(uint32_t*)&a, _b=*(uint32_t*)&b; uint32_t _r = (_a & 0x7FFFFFFF) | (~_b & 0x80000000); *(float*)&_r; })"),
    ("vfsgnjx.vv", 0b001010, "OPFVV", 32, "({ uint32_t _a=*(uint32_t*)&a, _b=*(uint32_t*)&b; uint32_t _r = _a ^ (_b & 0x80000000); *(float*)&_r; })"),
    ("vfmacc.vv",  0b101100, "OPFVV", 32, "(a + b * c)"),
    ("vfnmacc.vv", 0b101101, "OPFVV", 32, "(-(a + b * c))"),
    ("vfmsac.vv",  0b101110, "OPFVV", 32, "(b * c - a)"),
    ("vfnmsac.vv", 0b101111, "OPFVV", 32, "(-(b * c - a))"),
    # ── OPFVF float-scalar (a=vs2 vector, b=scalar from FPR) ──
    ("vfadd.vf",   0b000000, "OPFVF", 32, "(a + b)"),
    ("vfsub.vf",   0b000010, "OPFVF", 32, "(a - b)"),
    ("vfmul.vf",   0b100100, "OPFVF", 32, "(a * b)"),
    ("vfdiv.vf",   0b100000, "OPFVF", 32, "(a / b)"),
    # ── Float compare → mask ───────────────────────────────────
    ("vmfeq.vv",   0b011000, "OPFVV", 32, "((a == b) ? ~(uint32_t)0 : 0)"),
    ("vmfne.vv",   0b011001, "OPFVV", 32, "((a != b) ? ~(uint32_t)0 : 0)"),
    ("vmflt.vv",   0b011011, "OPFVV", 32, "((a < b) ? ~(uint32_t)0 : 0)"),
    ("vmfle.vv",   0b011101, "OPFVV", 32, "((a <= b) ? ~(uint32_t)0 : 0)"),
    ("vmfgt.vv",   0b011110, "OPFVV", 32, "((a > b) ? ~(uint32_t)0 : 0)"),
    ("vmfge.vv",   0b011111, "OPFVV", 32, "((a >= b) ? ~(uint32_t)0 : 0)"),
    # ── Float convert ──────────────────────────────────────────
    ("vfcvt.x.f.v",     0b100000, "OPFVV", 32, "((int32_t)a)"),
    ("vfcvt.xu.f.v",    0b100000, "OPFVV", 32, "((uint32_t)a)"),
    ("vfcvt.f.x.v",     0b100000, "OPFVV", 32, "((float)(int32_t)a)"),
    ("vfcvt.f.xu.v",    0b100000, "OPFVV", 32, "((float)(uint32_t)a)"),
    ("vfcvt.rtz.x.f.v", 0b100000, "OPFVV", 32, "((int32_t)truncf(a))"),
    # ── OPIVX extras ───────────────────────────────────────────
    ("vmin.vx",   0b000101, "OPIVX", 8|16|32, "(((T)(b) < sc) ? (T)(b) : sc)"),
    ("vminu.vx",  0b000100, "OPIVX", 8|16|32, "(((U)(b) < (U)(sc)) ? (U)(b) : (U)(sc))"),
    ("vmax.vx",   0b000111, "OPIVX", 8|16|32, "(((T)(b) > sc) ? (T)(b) : sc)"),
    ("vmaxu.vx",  0b000110, "OPIVX", 8|16|32, "(((U)(b) > (U)(sc)) ? (U)(b) : (U)(sc))"),
    ("vdiv.vx",   0b100000, "OPIVX", 8|16|32, "(sc ? (T)(b) / sc : (T)-1)"),
    ("vdivu.vx",  0b100001, "OPIVX", 8|16|32, "((U)(sc) ? (U)(b) / (U)(sc) : (U)-1)"),
    ("vrem.vx",   0b100010, "OPIVX", 8|16|32, "(sc ? (T)(b) % sc : (T)(b))"),
    ("vremu.vx",  0b100011, "OPIVX", 8|16|32, "((U)(sc) ? (U)(b) % (U)(sc) : (U)(b))"),
    # ── OPFVF float compare ────────────────────────────────────
    ("vmfeq.vf",  0b011000, "OPFVF", 32, "((a == b) ? ~(uint32_t)0 : 0)"),
    ("vmfne.vf",  0b011001, "OPFVF", 32, "((a != b) ? ~(uint32_t)0 : 0)"),
    ("vmflt.vf",  0b011011, "OPFVF", 32, "((a < b) ? ~(uint32_t)0 : 0)"),
    ("vmfle.vf",  0b011101, "OPFVF", 32, "((a <= b) ? ~(uint32_t)0 : 0)"),
    # ── Low-precision float convert (SEW=32, in-place conversion on VPR) ──
    ("vfcvt.bf16.s",  0b010000, "OPFVV", 32, "f32_to_bf16(*(uint32_t*)&a)"),
    ("vfcvt.s.bf16",  0b010000, "OPFVV", 32, "bf16_to_f32((uint16_t)*(uint32_t*)&a)"),
    ("vfcvt.e4m3.s",  0b010000, "OPFVV", 32, "f32_to_e4m3(*(uint32_t*)&a)"),
    ("vfcvt.s.e4m3",  0b010000, "OPFVV", 32, "e4m3_to_f32((uint8_t)*(uint32_t*)&a)"),
    ("vfcvt.e5m2.s",  0b010000, "OPFVV", 32, "f32_to_e5m2(*(uint32_t*)&a)"),
    ("vfcvt.s.e5m2",  0b010000, "OPFVV", 32, "e5m2_to_f32((uint8_t)*(uint32_t*)&a)"),
    # ── Reduction (handled specially in gen) ───────────────────
    ("vredsum.vs",   0b000000, "REDUCE_I", 8|16|32, "acc + (T)(b)"),
    ("vredmax.vs",   0b000111, "REDUCE_I", 8|16|32, "((T)(b) > acc ? (T)(b) : acc)"),
    ("vredmin.vs",   0b000101, "REDUCE_I", 8|16|32, "((T)(b) < acc ? (T)(b) : acc)"),
    ("vredand.vs",   0b000001, "REDUCE_U", 8|16|32, "(acc & (U)(b))"),
    ("vredor.vs",    0b000010, "REDUCE_U", 8|16|32, "(acc | (U)(b))"),
    ("vredxor.vs",   0b000011, "REDUCE_U", 8|16|32, "(acc ^ (U)(b))"),
    ("vfredosum.vs", 0b000011, "REDUCE_F", 32, "(acc + b)"),
    ("vfredusum.vs", 0b000001, "REDUCE_F", 32, "(acc + b)"),
    # ── Slide ──────────────────────────────────────────────────
    ("vslideup.vi",    0b001110, "SLIDE", 8|16|32, None),
    ("vslidedown.vi",  0b001111, "SLIDE", 8|16|32, None),
    ("vslide1up.vx",   0b001110, "SLIDE1", 8|16|32, None),
    ("vslide1down.vx", 0b001111, "SLIDE1", 8|16|32, None),
    # ── Gather ─────────────────────────────────────────────────
    ("vrgather.vv",     0b001100, "GATHER", 8|16|32, None),
    ("vrgatherei16.vv", 0b001110, "GATHER", 8|16|32, None),
    # ── vsetvli ─────────────────────────────────────────────────
    ("vsetvli", 0b000000, "VSETVLI", 32, None),
    # ── Unit-stride load/store ──────────────────────────────────
    ("vle32.v",  0b000000, "VLDST", 32, None),
    ("vse32.v",  0b000000, "VLDST", 32, None),
    # ── Mask logical ────────────────────────────────────────────
    ("vmand.mm",  0b011001, "MASK", 32, None),
    ("vmnand.mm", 0b011101, "MASK", 32, None),
    ("vmandn.mm", 0b011000, "MASK", 32, None),
    ("vmxor.mm",  0b001011, "MASK", 32, None),
    ("vmor.mm",   0b001010, "MASK", 32, None),
    ("vmcpy.m",   0b100100, "MASK", 32, None),
    ("vmclr.m",   0b100101, "MASK", 32, None),
    # ── vmerge ──────────────────────────────────────────────────
    ("vmerge.vvm", 0b010111, "VMERGE", 32, None),
]

F3 = {"OPIVV":0, "OPFVV":1, "OPMVV":2, "OPIVI":3, "OPIVX":4, "OPFVF":5, "OPMVX":6,
      "REDUCE_I":0, "REDUCE_U":0, "REDUCE_F":1, "SLIDE":3, "SLIDE1":4, "GATHER":0,
      "VSETVLI":7, "VLDST":0, "MASK":2, "VMERGE":0}
VM0_CATS = {"REDUCE_I", "REDUCE_U", "REDUCE_F", "VMERGE"}

# ═══════════════════════════════════════════════════════════════
#  Code generation
# ═══════════════════════════════════════════════════════════════

def vpr_offset(reg, sew_bytes):
    """Byte offset into VPR with LMUL support."""
    return f"((uint8_t*)vpr + ({reg} + _lmul_idx) * 128 + _lmul_lane * 4 + _lmul_sub * {sew_bytes})"

def load_store_vv(sew_bytes, reg, cast):
    """Generate load from VPR"""
    off = vpr_offset(reg, sew_bytes)
    return f"*(T*)({off})"


def gen_sew_handler(tag, sew, cat, body, sews=32):
    """Generate one SEW-specific handler"""
    is_float = (cat in ("OPFVV", "OPFVF"))
    is_widen = (sews == (8|16))  # EXACTLY 8+16, not 8+16+32
    is_narrow = is_widen  # narrowing has same sew pattern (8|16 only)
    if is_float:
        T = "float"
        U = "uint32_t"
    else:
        T = {8:"int8_t", 16:"int16_t", 32:"int32_t"}[sew]
        U = {8:"uint8_t", 16:"uint16_t", 32:"uint32_t"}[sew]
    W = {8:"int16_t", 16:"int32_t", 32:"int64_t"}[sew]  # wider type for widening ops
    WU = {8:"uint16_t", 16:"uint32_t", 32:"uint64_t"}[sew]
    sew_bytes = 4 if is_float else sew // 8
    phys_n = 1 if is_float else (4 // sew_bytes)  # sub-elements per 4-byte lane
    # _log_n = phys_n * LMUL (LMUL=1 always for now, set at engine_exec entry)
    is_vx = (cat == "OPIVX")
    is_vi = (cat == "OPIVI")
    is_vv = (cat in ("OPIVV", "OPFVV", "OPMVV"))
    is_vf = (cat == "OPFVF")

    lines = []
    lines.append(f"op_{tag}_e{sew}:")
    lines.append(f"{{  /* SEW={sew}, {phys_n} phys/lane × {1}=LMUL */")
    lines.append(f"    typedef {T} T __attribute__((unused));")
    lines.append(f"    typedef {U} U __attribute__((unused));")
    if is_widen or is_narrow:
        lines.append(f"    typedef {W} W __attribute__((unused));")
        lines.append(f"    typedef {WU} WU __attribute__((unused));")
    lines.append(f"    enum {{ SEW = {sew} }};")
    lines.append(f"    int vd = ip[-1].rd;")
    lines.append(f"    int vs1 = ip[-1].rs1;")
    lines.append(f"    int vs2 = ip[-1].rs2;")
    lines.append(f"    uint32_t _vm = (ip[-1].inst >> 25) & 1;")

    # LMUL: _log_n = phys_n * _lmul elements per lane. _lmul=1 for now.
    lines.append(f"    int _phys_n = {phys_n};")
    lines.append(f"    int _log_n = _phys_n * _lmul;")
    lines.append(f"    int _phys_elems = 32 * _phys_n;  /* elements per physical register */")
    lines.append(f"    FOR_EACH_LANE {{")
    lines.append(f"        for (uint32_t _e = 0; _e < (uint32_t)_log_n && (_li * _log_n + _e) < _vl; _e++) {{")
    lines.append(f"            int _gi = _li * _log_n + _e;  /* global element index */")
    lines.append(f"            int _lmul_idx = _gi / _phys_elems;")
    lines.append(f"            int _lmul_rem = _gi % _phys_elems;")
    lines.append(f"            int _lmul_lane = _lmul_rem / _phys_n;")
    lines.append(f"            int _lmul_sub  = _lmul_rem % _phys_n;")
    # v0 mask: check bit 0 of element at lane 0
    lines.append(f"            int _mask_bit = *(uint8_t*)((uint8_t*)vpr + 0*128 + _li*4 + _e);")
    lines.append(f"            if (!_vm && !(_mask_bit & 1)) continue;")

    # Load operands
    if is_vv:
        lines.append(f"            T a = {load_store_vv(sew_bytes, 'vs1', T)};")
        lines.append(f"            T b = {load_store_vv(sew_bytes, 'vs2', T)};")
        if cat == "OPMVV":
            # OPMVV has vd = vs1* vs2 + vd, so need c=old vd
            lines.append(f"            T c = {load_store_vv(sew_bytes, 'vd', T)};")
    elif is_vx:
        lines.append(f"            T b = {load_store_vv(sew_bytes, 'vs2', T)};")
        lines.append(f"            T sc = (T)GPR(vs1, _li);")
    elif is_vf:
        # OPFVF: a = vs2 (vector), b = scalar from FPR(rs1, 0)
        lines.append(f"            T a = {load_store_vv(sew_bytes, 'vs2', T)};")
        lines.append(f"            T b = FR(vs1, 0);  /* scalar FPR */")
    elif is_vi:
        # OPIVI: b = vs2 (vector), sc = 5-bit sign-extended immediate
        lines.append(f"    int32_t _imm = ((int32_t)((ip[-1].inst >> 15) & 0x1F) << 27) >> 27;")
        lines.append(f"            T b = {load_store_vv(sew_bytes, 'vs2', T)};")
        lines.append(f"            T sc = (T)_imm;")

    # NO_HANDLER categories: dispatch+customasm from generator, handler hand-written in engine.c
    NO_HANDLER = {"REDUCE_I", "REDUCE_U", "REDUCE_F", "SLIDE", "SLIDE1", "GATHER"}
    if cat in NO_HANDLER or is_widen or is_narrow:
        return None  # skip handler generation entirely
        lines.append("    /* (hand-written handler, not auto-generated) */")
        lines.append("    NEXT();")
        lines.append("}")
        return '\n'.join(lines)

    expr = body
    # Write result
    if is_vv or is_vx or is_vf:
        lines.append(f"            T _r = {expr};")
        lines.append(f"            {load_store_vv(sew_bytes, 'vd', T)} = _r;")

    lines.append(f"        }}")
    lines.append(f"        PC(_li) += 4;")
    lines.append(f"    }}")
    lines.append(f"    NEXT();")
    lines.append(f"}}")
    return '\n'.join(lines)

def gen_dispatch_pattern(tag, f6, cat):
    f3 = F3[cat]
    vm = '0' if cat in VM0_CATS else '?'
    return f'X({tag}, "{f6:06b}{vm} ????? ????? {f3:03b} ????? 10101 11", TYPE_R, imm0);'

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

HANDLER_CODE = {}
def _h(tag, code):
    HANDLER_CODE[tag] = code.replace('\n',' ').strip()

_h("vredsum_vs",   "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;int32_t acc=0;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc+=*(int32_t*)&VR(vs2,_li);PC(_li)+=4;}*(int32_t*)&VR(vd,0)=acc;NEXT();")
_h("vredmax_vs",   "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;int32_t acc=INT32_MIN;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){int32_t v=*(int32_t*)&VR(vs2,_li);if(v>acc)acc=v;}PC(_li)+=4;}*(int32_t*)&VR(vd,0)=acc;NEXT();")
_h("vredmin_vs",   "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;int32_t acc=INT32_MAX;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){int32_t v=*(int32_t*)&VR(vs2,_li);if(v<acc)acc=v;}PC(_li)+=4;}*(int32_t*)&VR(vd,0)=acc;NEXT();")
_h("vredand_vs",   "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;uint32_t acc=~0U;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc&=*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}*(uint32_t*)&VR(vd,0)=acc;NEXT();")
_h("vredor_vs",    "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;uint32_t acc=0;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc|=*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}*(uint32_t*)&VR(vd,0)=acc;NEXT();")
_h("vredxor_vs",   "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;uint32_t acc=0;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc^=*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}*(uint32_t*)&VR(vd,0)=acc;NEXT();")
_h("vfredosum_vs", "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;float acc=0.0f;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc+=VR(vs2,_li);PC(_li)+=4;}VR(vd,0)=acc;NEXT();")
_h("vfredusum_vs", "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;float acc=0.0f;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))acc+=VR(vs2,_li);PC(_li)+=4;}VR(vd,0)=acc;NEXT();")
_h("vslideup_vi",    "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1,uimm=(ip[-1].inst>>15)&0x1F;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){if(_li>=uimm)VR(vd,_li)=VR(vs2,_li-uimm);}PC(_li)+=4;}NEXT();")
_h("vslidedown_vi",  "int vd=ip[-1].rd,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1,uimm=(ip[-1].inst>>15)&0x1F;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){if(_li+uimm<_vl)VR(vd,_li)=VR(vs2,_li+uimm);}PC(_li)+=4;}NEXT();")
_h("vslide1up_vx",   "int vd=ip[-1].rd,rs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))VR(vd,_li)=(_li==0)?*(float*)&GPR(rs1,0):VR(vs2,_li-1);PC(_li)+=4;}NEXT();")
_h("vslide1down_vx", "int vd=ip[-1].rd,rs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))VR(vd,_li)=(_li+1<_vl)?VR(vs2,_li+1):*(float*)&GPR(rs1,0);PC(_li)+=4;}NEXT();")
_h("vrgather_vv",     "int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){uint32_t idx=*(uint32_t*)&VR(vs2,_li);VR(vd,_li)=(idx<_vl)?VR(vs1,idx):0.0f;}PC(_li)+=4;}NEXT();")
_h("vrgatherei16_vv", "int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){uint32_t idx=*(uint16_t*)&VR(vs2,_li);VR(vd,_li)=(idx<_vl)?VR(vs1,idx):0.0f;}PC(_li)+=4;}NEXT();")
_h("vwadd_vv",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(int32_t*)&VR(vd,_li)=(int32_t)*(int16_t*)&VR(vs1,_li)+(int32_t)*(int16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwaddu_vv",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=(uint32_t)*(uint16_t*)&VR(vs1,_li)+(uint32_t)*(uint16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwsub_vv",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(int32_t*)&VR(vd,_li)=(int32_t)*(int16_t*)&VR(vs1,_li)-(int32_t)*(int16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwsubu_vv",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=(uint32_t)*(uint16_t*)&VR(vs1,_li)-(uint32_t)*(uint16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwmul_vv",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(int32_t*)&VR(vd,_li)=(int32_t)*(int16_t*)&VR(vs1,_li)*(int32_t)*(int16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwmulu_vv",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=(uint32_t)*(uint16_t*)&VR(vs1,_li)*(uint32_t)*(uint16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwmulsu_vv", "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(int64_t*)&VR(vd,_li)=(int64_t)*(int16_t*)&VR(vs1,_li)*(uint64_t)*(uint16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vwmacc_vv",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(int32_t*)&VR(vd,_li)+=(int32_t)*(int16_t*)&VR(vs1,_li)*(int32_t)*(int16_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vnsrl_wv",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){uint32_t a=*(uint32_t*)&VR(vs1,_li),b=*(uint32_t*)&VR(vs2,_li);*(uint16_t*)&VR(vd,_li)=(uint16_t)(a>>(b&31));}PC(_li)+=4;}NEXT();}")
_h("vnsra_wv",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){int32_t a=*(int32_t*)&VR(vs1,_li);uint32_t b=*(uint32_t*)&VR(vs2,_li);*(int16_t*)&VR(vd,_li)=(int16_t)(a>>(b&31));}PC(_li)+=4;}NEXT();}")
_h("vsetvli",    "int rd=ip[-1].rd,rs1=ip[-1].rs1;uint32_t avl=GPR(rs1,0),new_vl=avl?(avl<32?avl:32):32;if(vl)*vl=new_vl;_vl=new_vl;uint32_t vtypei=(ip[-1].inst>>20)&0x7FF;static const uint8_t sm[8]={8,16,32,64};static const uint8_t lm[8]={1,2,4,8};uint32_t vs=sm[(vtypei>>0)&7],vl2=lm[(vtypei>>3)&7];if(!vs)vs=32;if(!vl2)vl2=1;if(sew)*sew=vs;if(lmul)*lmul=vl2;_sew=vs;_lmul=vl2;FOR_EACH_LANE{GPR(rd,_li)=new_vl;PC(_li)+=4;}NEXT();")
_h("vle32_v",    "int vd=ip[-1].rd,rs1=ip[-1].rs1;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){uint32_t a=GPR(rs1,_li)+_li*4;VR(vd,_li)=(a+4<=s->vram_size)?*(float*)(s->vram_ptr+a):0.0f;}PC(_li)+=4;}NEXT();")
_h("vse32_v",    "int vs3=ip[-1].rd,rs1=ip[-1].rs1;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1))){uint32_t a=GPR(rs1,_li)+_li*4;if(a+4<=s->vram_size)*(float*)(s->vram_ptr+a)=VR(vs3,_li);}PC(_li)+=4;}NEXT();")
_h("vmand_mm",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=*(uint32_t*)&VR(vs1,_li)&*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vmnand_mm",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=~(*(uint32_t*)&VR(vs1,_li)&*(uint32_t*)&VR(vs2,_li));PC(_li)+=4;}NEXT();}")
_h("vmandn_mm",  "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=*(uint32_t*)&VR(vs1,_li)&~*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vmxor_mm",   "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=*(uint32_t*)&VR(vs1,_li)^*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vmor_mm",    "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;uint32_t vm=(ip[-1].inst>>25)&1;FOR_EACH_LANE{if((uint32_t)_li<_vl&&(vm||(*(uint32_t*)&VR(0,_li)&1)))*(uint32_t*)&VR(vd,_li)=*(uint32_t*)&VR(vs1,_li)|*(uint32_t*)&VR(vs2,_li);PC(_li)+=4;}NEXT();}")
_h("vmcpy_m",    "{int vd=ip[-1].rd,vs1=ip[-1].rs1;FOR_EACH_LANE{if((uint32_t)_li<_vl)*(uint32_t*)&VR(vd,_li)=*(uint32_t*)&VR(vs1,_li);PC(_li)+=4;}NEXT();}")
_h("vmclr_m",    "{int vd=ip[-1].rd;FOR_EACH_LANE{if((uint32_t)_li<_vl)*(uint32_t*)&VR(vd,_li)=0;PC(_li)+=4;}NEXT();}")
_h("vmerge_vvm", "{int vd=ip[-1].rd,vs1=ip[-1].rs1,vs2=ip[-1].rs2;FOR_EACH_LANE{if((uint32_t)_li<_vl)VR(vd,_li)=(*(uint32_t*)&VR(0,_li)&1)?VR(vs2,_li):VR(vs1,_li);PC(_li)+=4;}NEXT();}")

def generate():
    dispatch = []
    handlers = []
    cusmasm = []

    dispatch.append(f"/* AUTO-GENERATED by {SCRIPT} */")
    handlers.append(f"/* AUTO-GENERATED by {SCRIPT} */")
    handlers.append(f"/* Paste into engine.c at the RVV section */")
    handlers.append("")
    # Saturation macros used by generated handlers
    handlers.append("/* Saturation helpers */")
    handlers.append("#define RV_SAT_ADD_S(a, b) ({ T _r_; "
                    "if (((b)>0) && ((a)>(T)((T)-1-(b)))) _r_ = ~(T)((T)-1 << (SEW-1)); "
                    "else if (((b)<0) && ((a)<(T)((T)-1-(b)))) _r_ = (T)((T)-1 << (SEW-1)); "
                    "else _r_ = (T)(a)+(T)(b); _r_; })")
    handlers.append("#define RV_SAT_ADD_U(a, b) ({ T _r_ = (T)(a)+(T)(b); _r_ < (a) ? (T)-1 : _r_; })")
    handlers.append("#define RV_SAT_SUB_S(a, b) ({ T _r_; "
                    "if (((b)<0) && ((a)>(T)((T)-1+(b)))) _r_ = ~(T)((T)-1 << (SEW-1)); "
                    "else if (((b)>0) && ((a)<(T)((T)-1+(b)))) _r_ = (T)((T)-1 << (SEW-1)); "
                    "else _r_ = (T)(a)-(T)(b); _r_; })")
    handlers.append("#define RV_SAT_SUB_U(a, b) ({ T _r_ = (T)(a)-(T)(b); _r_ > (a) ? (T)0 : _r_; })")
    handlers.append("")

    # Predeclare all _e32 handlers (they're at the end, but goto needs them forward-declared? No, computed-goto labels are function-scope, fine.)

    for name, f6, cat, sews, body in INSTRUCTIONS:
        tag = name.replace('.', '_')

        # dispatch.h
        dispatch.append("    " + gen_dispatch_pattern(tag, f6, cat) + " \\")

        # engine.c: use hand-written code if available, else auto-generate
        hcode = HANDLER_CODE.get(tag)
        if hcode:
            handlers.append(f"op_{tag}:")
            handlers.append(hcode.strip())
            handlers.append("")
            continue

        # Auto-generate: dispatch stub + SEW variants
        needs_stub = ((sews & (8|16)) and (sews & 32)) and (cat not in VM0_CATS) and (cat not in ("REDUCE_I","REDUCE_U","REDUCE_F","SLIDE","SLIDE1","GATHER"))
        if needs_stub:
            handlers.append(f"op_{tag}:")
            if sews & 8:  handlers.append(f"    if (_sew == 8)  goto op_{tag}_e8;")
            if sews & 16: handlers.append(f"    if (_sew == 16) goto op_{tag}_e16;")
            handlers.append(f"    goto op_{tag}_e32;")
        else:
            handlers.append(f"op_{tag}:")
        h32 = gen_sew_handler(tag, 32, cat, body, sews)
        if h32:
            handlers.append(h32.replace(f"op_{tag}_e32", f"op_{tag}", 1))
        for sew in [8, 16]:
            if sews & sew:
                h = gen_sew_handler(tag, sew, cat, body, sews)
                if h: handlers.append(h)
        handlers.append("")

        # customasm
        f3 = F3[cat]
        if cat in ("OPIVV", "OPFVV", "OPMVV"):
            cusmasm.append(f"    '{name}': ({f6:#08b}, {f3}, 'RVV'),")
        elif cat == "OPIVX":
            cusmasm.append(f"    '{name}': ({f6:#08b}, {f3}, 'RVVX'),")
        elif cat == "OPFVF":
            cusmasm.append(f"    '{name}': ({f6:#08b}, {f3}, 'RVVF'),")

    # Write dispatch patterns
    with open(OUT_DISPATCH, 'w') as f:
        f.write(f"/* AUTO-GENERATED by {SCRIPT} — include in dispatch.h INSTRUCTION_LIST */\n")
        for d in dispatch:
            f.write(d + "\n")
    # Write handlers
    with open(OUT_HANDLERS, 'w') as f:
        f.write(f"/* AUTO-GENERATED by {SCRIPT} — include in engine.c */\n")
        f.write('\n'.join(handlers))
        f.write("\n")
    # Print customasm entries to stdout for manual copy
    print(f"Generated {OUT_DISPATCH} + {OUT_HANDLERS}: {len(INSTRUCTIONS)} instructions × 3 SEW = {len(INSTRUCTIONS)*3} handlers")
    print("=== customasm.py entries ===")
    for c in cusmasm: print(c)

if __name__ == '__main__':
    generate()
