const hex = (v) => "0x" + (v >>> 0).toString(16);
const i32 = (v) => v | 0;

function fmtFloat(bits) {
    const arr = new Uint32Array(1);
    arr[0] = bits;
    const f = new Float32Array(arr.buffer)[0];
    return `${hex(bits)} (${f})`;
}

function fmtArgs(labels, ops) {
    const parts = [];
    for (let i = 0; i < labels.length; i++) {
        parts.push(labels[i] + "=" + hex(ops[i]));
    }
    return parts.join(", ");
}

/* ── Instruction format table (level=0) ── */

const INST_FMT = {
    /* branches: src1, src2, imm, pc */
    3:  (rec) => { const o=rec.operands; return `[BEQ] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${o[0]===o[1]}`; },
    4:  (rec) => { const o=rec.operands; return `[BNE] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${o[0]!==o[1]}`; },
    5:  (rec) => { const o=rec.operands; return `[BLT] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${i32(o[0])<i32(o[1])}`; },
    6:  (rec) => { const o=rec.operands; return `[BGE] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${i32(o[0])>=i32(o[1])}`; },
    7:  (rec) => { const o=rec.operands; return `[BLTU] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${o[0]<o[1]}`; },
    8:  (rec) => { const o=rec.operands; return `[BGEU] src1=${hex(o[0])}, src2=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])}, taken=${o[0]>=o[1]}`; },

    /* jumps */
    1:  (rec) => { const o=rec.operands; const tgt=(o[2]+o[1]-4)>>>0; return `[JAL] rd=${o[0]}, imm=${hex(o[1])}, pc=${hex(o[2])} -> ${hex(tgt)}`; },
    2:  (rec) => { const o=rec.operands; const tgt=((o[1]+o[2])&~1)>>>0; return `[JALR] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, pc=${hex(o[3])} -> ${hex(tgt)}`; },

    /* loads: rd, addr (no val — not in trace) */
    9:  (rec) => { const o=rec.operands; return `[LB] rd=${o[0]}, addr=${hex(o[1])}`; },
    10: (rec) => { const o=rec.operands; return `[LH] rd=${o[0]}, addr=${hex(o[1])}`; },
    11: (rec) => { const o=rec.operands; return `[LW] rd=${o[0]}, addr=${hex(o[1])}`; },
    12: (rec) => { const o=rec.operands; return `[LBU] rd=${o[0]}, addr=${hex(o[1])}`; },
    13: (rec) => { const o=rec.operands; return `[LHU] rd=${o[0]}, addr=${hex(o[1])}`; },

    /* stores: addr, val */
    14: (rec) => { const o=rec.operands; return `[SB] addr=${hex(o[0])}, val=${hex(o[1]&0xFF)}`; },
    15: (rec) => { const o=rec.operands; return `[SH] addr=${hex(o[0])}, val=${hex(o[1]&0xFFFF)}`; },
    16: (rec) => { const o=rec.operands; return `[SW] addr=${hex(o[0])}, val=${hex(o[1])}`; },

    /* U-type */
    17: (rec) => { const o=rec.operands; return `[LUI] rd=${o[0]}, imm=${hex(o[1])}`; },
    18: (rec) => { const o=rec.operands; const r=(o[2]+o[1])>>>0; return `[AUIPC] rd=${o[0]}, imm=${hex(o[1])}, pc=${hex(o[2])}, result=${hex(r)}`; },

    /* I-type ALU: rd, src1, imm → result computable */
    19: (rec) => { const o=rec.operands; const r=(o[1]+o[2])>>>0; return `[ADDI] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${hex(r)}`; },
    20: (rec) => { const o=rec.operands; const r=i32(o[1])<i32(o[2])?1:0; return `[SLTI] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${r}`; },
    21: (rec) => { const o=rec.operands; const r=o[1]<o[2]>>>0?1:0; return `[SLTIU] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${r}`; },
    22: (rec) => { const o=rec.operands; const r=(o[1]^o[2])>>>0; return `[XORI] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${hex(r)}`; },
    23: (rec) => { const o=rec.operands; const r=(o[1]|o[2])>>>0; return `[ORI] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${hex(r)}`; },
    24: (rec) => { const o=rec.operands; const r=(o[1]&o[2])>>>0; return `[ANDI] rd=${o[0]}, src1=${hex(o[1])}, imm=${hex(o[2])}, result=${hex(r)}`; },

    /* I-type shift: rd, src1, shamt */
    25: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=(o[1]<<sh)>>>0; return `[SLLI] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },
    26: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=o[1]>>>sh; return `[SRLI] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },
    27: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=(i32(o[1])>>sh)>>>0; return `[SRAI] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },

    /* R-type ALU: rd, src1, src2 */
    28: (rec) => { const o=rec.operands; const r=(o[1]+o[2])>>>0; return `[ADD] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    29: (rec) => { const o=rec.operands; const r=(o[1]-o[2])>>>0; return `[SUB] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    30: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=(o[1]<<sh)>>>0; return `[SLL] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },
    31: (rec) => { const o=rec.operands; const r=i32(o[1])<i32(o[2])?1:0; return `[SLT] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${r}`; },
    32: (rec) => { const o=rec.operands; const r=o[1]<o[2]?1:0; return `[SLTU] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${r}`; },
    33: (rec) => { const o=rec.operands; const r=(o[1]^o[2])>>>0; return `[XOR] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    34: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=o[1]>>>sh; return `[SRL] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },
    35: (rec) => { const o=rec.operands; const sh=o[2]&0x1F; const r=(i32(o[1])>>sh)>>>0; return `[SRA] rd=${o[0]}, src1=${hex(o[1])}, shamt=${sh}, result=${hex(r)}`; },
    36: (rec) => { const o=rec.operands; const r=(o[1]|o[2])>>>0; return `[OR] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    37: (rec) => { const o=rec.operands; const r=(o[1]&o[2])>>>0; return `[AND] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },

    /* M-ext: rd, src1, src2 */
    38: (rec) => { const o=rec.operands; const r=(o[1]*o[2])>>>0; return `[MUL] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    39: (rec) => { const o=rec.operands; const r=Number((BigInt(i32(o[1]))*BigInt(i32(o[2])))>>32n)&0xFFFFFFFF; return `[MULH] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    40: (rec) => { const o=rec.operands; const r=Number((BigInt(i32(o[1]))*BigInt(o[2]))>>32n)&0xFFFFFFFF; return `[MULHSU] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    41: (rec) => { const o=rec.operands; const r=Number((BigInt(o[1])*BigInt(o[2]))>>32n)&0xFFFFFFFF; return `[MULHU] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    42: (rec) => { const o=rec.operands; const r=o[2]===0?-1:i32(o[1])/i32(o[2]); return `[DIV] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    43: (rec) => { const o=rec.operands; const r=o[2]===0?0xFFFFFFFF:(o[1]/o[2])>>>0; return `[DIVU] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    44: (rec) => { const o=rec.operands; const r=o[2]===0?o[1]:i32(o[1])%i32(o[2]); return `[REM] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },
    45: (rec) => { const o=rec.operands; const r=o[2]===0?o[1]:(o[1]%o[2])>>>0; return `[REMU] rd=${o[0]}, src1=${hex(o[1])}, src2=${hex(o[2])}, result=${hex(r)}`; },

    /* CSR: rd, csr, src1/uimm */
    46: (rec) => { const o=rec.operands; return `[CSRRW] rd=${o[0]}, csr=${hex(o[1])}, src1=${hex(o[2])}`; },
    47: (rec) => { const o=rec.operands; return `[CSRRS] rd=${o[0]}, csr=${hex(o[1])}, src1=${hex(o[2])}`; },
    48: (rec) => { const o=rec.operands; return `[CSRRC] rd=${o[0]}, csr=${hex(o[1])}, src1=${hex(o[2])}`; },
    49: (rec) => { const o=rec.operands; return `[CSRRWI] rd=${o[0]}, csr=${hex(o[1])}, uimm=${o[2]}`; },
    50: (rec) => { const o=rec.operands; return `[CSRRSI] rd=${o[0]}, csr=${hex(o[1])}, uimm=${o[2]}`; },
    51: (rec) => { const o=rec.operands; return `[CSRRCI] rd=${o[0]}, csr=${hex(o[1])}, imm=${hex(o[2])}`; },

    /* system */
    52: () => "[ECALL] environment call",
    53: () => "[EBREAK] breakpoint instruction executed",

    /* FP loads/stores */
    54: (rec) => { const o=rec.operands; return `[FLW] rd=${o[0]}, addr=${hex(o[1])}`; },
    55: (rec) => { const o=rec.operands; return `[FSW] addr=${hex(o[0])}, val=${fmtFloat(o[1])}`; },

    /* FP fused-mac: rd, rs1, rs2, rs3 */
    56: (rec) => { const o=rec.operands; return `[FMADD_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}, rs3=${fmtFloat(o[3])}`; },
    57: (rec) => { const o=rec.operands; return `[FMSUB_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}, rs3=${fmtFloat(o[3])}`; },
    58: (rec) => { const o=rec.operands; return `[FNMSUB_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}, rs3=${fmtFloat(o[3])}`; },
    59: (rec) => { const o=rec.operands; return `[FNMADD_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}, rs3=${fmtFloat(o[3])}`; },

    /* FP arithmetic */
    60: (rec) => { const o=rec.operands; return `[FADD_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    61: (rec) => { const o=rec.operands; return `[FSUB_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    62: (rec) => { const o=rec.operands; return `[FMUL_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    63: (rec) => { const o=rec.operands; return `[FDIV_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    64: (rec) => { const o=rec.operands; return `[FSQRT_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },

    /* FP sign-inject */
    65: (rec) => { const o=rec.operands; return `[FSGNJ_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    66: (rec) => { const o=rec.operands; return `[FSGNJN_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    67: (rec) => { const o=rec.operands; return `[FSGNJX_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },

    /* FP min/max */
    68: (rec) => { const o=rec.operands; return `[FMIN_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    69: (rec) => { const o=rec.operands; return `[FMAX_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },

    /* FP convert */
    70: (rec) => { const o=rec.operands; return `[FCVT_W_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    71: (rec) => { const o=rec.operands; return `[FCVT_WU_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    72: (rec) => { const o=rec.operands; return `[FMV_X_W] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    73: (rec) => { const o=rec.operands; return `[FEQ_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    74: (rec) => { const o=rec.operands; return `[FLT_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    75: (rec) => { const o=rec.operands; return `[FLE_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}, rs2=${fmtFloat(o[2])}`; },
    76: (rec) => { const o=rec.operands; return `[FCLASS_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    77: (rec) => { const o=rec.operands; return `[FCVT_S_W] rd=${o[0]}, rs1=${hex(o[1])}`; },
    78: (rec) => { const o=rec.operands; return `[FCVT_S_WU] rd=${o[0]}, rs1=${hex(o[1])}`; },
    79: (rec) => { const o=rec.operands; return `[FMV_W_X] rd=${o[0]}, rs1=${hex(o[1])}`; },

    /* custom float converts: rd, rs1 */
    80: (rec) => { const o=rec.operands; return `[FCVT_S_BF16] rd=${o[0]}, rs1=${hex(o[1]&0xFFFF)}`; },
    81: (rec) => { const o=rec.operands; return `[FCVT_BF16_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    82: (rec) => { const o=rec.operands; return `[FCVT_S_E4M3] rd=${o[0]}, rs1=${hex(o[1]&0xFF)}`; },
    83: (rec) => { const o=rec.operands; return `[FCVT_E4M3_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    84: (rec) => { const o=rec.operands; return `[FCVT_S_E5M2] rd=${o[0]}, rs1=${hex(o[1]&0xFF)}`; },
    85: (rec) => { const o=rec.operands; return `[FCVT_E5M2_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    86: (rec) => { const o=rec.operands; return `[FCVT_S_E2M1] rd=${o[0]}, rs1=${hex(o[1]&0xFF)}`; },
    87: (rec) => { const o=rec.operands; return `[FCVT_E2M1_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },

    /* custom math: rd, rs1 */
    88: (rec) => { const o=rec.operands; return `[FEXP_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    89: (rec) => { const o=rec.operands; return `[FLN_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    90: (rec) => { const o=rec.operands; return `[FRCP_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    91: (rec) => { const o=rec.operands; return `[FRSQRT_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    92: (rec) => { const o=rec.operands; return `[FTANH_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    93: (rec) => { const o=rec.operands; return `[FSIGMOID_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    94: (rec) => { const o=rec.operands; return `[FSIN_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
    95: (rec) => { const o=rec.operands; return `[FCOS_S] rd=${o[0]}, rs1=${fmtFloat(o[1])}`; },
};

/* ── Event format table (level=1) ── */

const EVT_FMT = {
    1:  (rec) => { const o=rec.operands; return `REG_WRITE  offset=${hex(o[0])}, value=${hex(o[1])}`; },
    2:  (rec) => { const o=rec.operands; return `REG_READ   offset=${hex(o[0])}, value=${hex(o[1])}`; },
    3:  (rec) => { const o=rec.operands; return `DMA_START  src=0x${o[1].toString(16)}${o[0].toString(16).padStart(8,'0')}, dst=0x${o[3].toString(16)}${o[2].toString(16).padStart(8,'0')}, size=${hex(o[4])}`; },
    4:  (rec) => { const o=rec.operands; const s=["OK","BUSY","COMPLETE","","ERROR"][o[0]]||o[0]; return `DMA_COMPLETE  status=${s}`; },
    5:  (rec) => { const o=rec.operands; return `KERNEL_DISPATCH  addr=${hex(o[0])}, grid=(${o[1]},${o[2]},${o[3]}), block=(${o[4]},${o[5]},${o[6]})`; },
    6:  (rec) => { const o=rec.operands; return `KERNEL_COMPLETE  status=${o[0]}`; },
    7:  (rec) => { const o=rec.operands; const t=["KERNEL_DONE","DMA_DONE","ERROR"][o[0]]||o[0]; return `IRQ_FIRE  type=${t}, vector=${o[1]}`; },
    8:  (rec) => { const o=rec.operands; return `ERROR  code=${hex(o[0])}, detail=${hex(o[1])}`; },
    9:  (rec) => { const o=rec.operands; const st=["READY","BUSY","ERROR"][o[0]]||o[0]; const sn=["READY","BUSY","ERROR"][o[1]]||o[1]; return `STATE_CHANGE  ${st} -> ${sn}`; },
};

/* ── Public API ── */

export function formatRecord(rec) {
    if (rec.level === 1) {
        const fn = EVT_FMT[rec.opcode];
        return fn ? fn(rec) : `EVENT_${rec.opcode} ` + rec.operands.map(hex).join(" ");
    }

    const fn = INST_FMT[rec.opcode];
    return fn ? fn(rec) : `[OP_${rec.opcode}] ` + rec.operands.map(hex).join(" ");
}
