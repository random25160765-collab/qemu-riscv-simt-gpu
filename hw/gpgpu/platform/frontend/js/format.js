const EVENT_TABLE = {
    1:  { name: "REG_WRITE",       nargs: 2 },
    2:  { name: "REG_READ",        nargs: 2 },
    3:  { name: "DMA_START",       nargs: 5 },
    4:  { name: "DMA_COMPLETE",    nargs: 1 },
    5:  { name: "KERNEL_DISPATCH", nargs: 7 },
    6:  { name: "KERNEL_COMPLETE", nargs: 1 },
    7:  { name: "IRQ_FIRE",        nargs: 2 },
    8:  { name: "ERROR_EVENT",     nargs: 2 },
    9:  { name: "STATE_CHANGE",    nargs: 2 },
};

const INST_TABLE = {
    1:  "JAL",   2:  "JALR",  3:  "BEQ",   4:  "BNE",
    5:  "BLT",   6:  "BGE",   7:  "BLTU",  8:  "BGEU",
    9:  "LB",    10: "LH",    11: "LW",    12: "LBU",
    13: "LHU",   14: "SB",    15: "SH",    16: "SW",
    17: "ADDI",  18: "SLTI",  19: "SLTIU", 20: "XORI",
    21: "ORI",   22: "ANDI",  23: "SLLI",  24: "SRLI",
    25: "SRAI",  26: "ADD",   27: "SUB",   28: "MUL",
    29: "SLT",   30: "SLTU",  31: "XOR",   32: "DIV",
    33: "DIVU",  34: "REM",   35: "REMU",  36: "SLL",
    37: "SRL",   38: "SRA",   39: "OR",    40: "AND",
    41: "MULH",  42: "MULHU", 43: "MULHSU",44: "CSRRW",
    45: "CSRRS", 46: "CSRRC", 47: "CSRRWI",48: "CSRRSI",
    49: "CSRRCI",50: "LUI",   51: "AUIPC", 52: "FENCE",
    53: "EBREAK",54: "FLW",   55: "FSW",   56: "FMADD_S",
    57: "FMSUB_S",58: "FNMSUB_S",59: "FNMADD_S",60: "FADD_S",
    61: "FSUB_S",62: "FMUL_S",63: "FDIV_S",64: "FSQRT_S",
    65: "FSGNJ_S",66: "FSGNJN_S",67: "FSGNJX_S",68: "FMIN_S",
    69: "FMAX_S",70: "FCVT_W_S",71: "FCVT_WU_S",72: "FMV_X_W",
    73: "FEQ_S", 74: "FLT_S", 75: "FLE_S", 76: "FCLASS_S",
    77: "FCVT_S_W",78: "FCVT_S_WU",79: "FMV_W_X",80: "FLPADD",
    81: "FLPSUB",82: "FLPMUL",83: "FLPDIV",84: "FLPSQRT",
    85: "FLPEXP",86: "FLPLN", 87: "FLPRCP",88: "FLPRSQRT",
    89: "FLPTANH",90: "FLPSIGMOID",91: "FLPSIN",92: "FLPCOS",
    93: "FLPFMADD",94: "FLPFMSUB",95: "FLPFNMADD",
};

export function formatRecord(rec) {
    const ops = rec.operands || [];

    if (rec.level === 1) {
        const info = EVENT_TABLE[rec.opcode];
        const name = info ? info.name : `EVENT_${rec.opcode}`;
        const vals = ops.map(v => `0x${(v >>> 0).toString(16)}`).join(" ");
        return `[${name}] ${vals}`;
    }

    const name = INST_TABLE[rec.opcode] || `OP_${rec.opcode}`;
    const vals = ops.map(v => `0x${(v >>> 0).toString(16)}`).join(" ");
    return `[${name}] ${vals}`;
}
