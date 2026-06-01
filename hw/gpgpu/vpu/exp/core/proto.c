/*
 * 观测协议 — 独立版：trace/event 直接输出到 stderr
 */
#include "proto.h"

/* 指令名称查找表 */
static const char *inst_name(uint32_t code)
{
    switch (code & 0x00FFFFFF) {
    case INST_JAL & 0x00FFFFFF: return "JAL";
    case INST_JALR & 0x00FFFFFF: return "JALR";
    case INST_BEQ & 0x00FFFFFF: return "BEQ";
    case INST_BNE & 0x00FFFFFF: return "BNE";
    case INST_BLT & 0x00FFFFFF: return "BLT";
    case INST_BGE & 0x00FFFFFF: return "BGE";
    case INST_BLTU & 0x00FFFFFF: return "BLTU";
    case INST_BGEU & 0x00FFFFFF: return "BGEU";
    case INST_LB & 0x00FFFFFF: return "LB";
    case INST_LH & 0x00FFFFFF: return "LH";
    case INST_LW & 0x00FFFFFF: return "LW";
    case INST_LBU & 0x00FFFFFF: return "LBU";
    case INST_LHU & 0x00FFFFFF: return "LHU";
    case INST_SB & 0x00FFFFFF: return "SB";
    case INST_SH & 0x00FFFFFF: return "SH";
    case INST_SW & 0x00FFFFFF: return "SW";
    case INST_LUI & 0x00FFFFFF: return "LUI";
    case INST_AUIPC & 0x00FFFFFF: return "AUIPC";
    case INST_ADDI & 0x00FFFFFF: return "ADDI";
    case INST_SLTI & 0x00FFFFFF: return "SLTI";
    case INST_SLTIU & 0x00FFFFFF: return "SLTIU";
    case INST_XORI & 0x00FFFFFF: return "XORI";
    case INST_ORI & 0x00FFFFFF: return "ORI";
    case INST_ANDI & 0x00FFFFFF: return "ANDI";
    case INST_SLLI & 0x00FFFFFF: return "SLLI";
    case INST_SRLI & 0x00FFFFFF: return "SRLI";
    case INST_SRAI & 0x00FFFFFF: return "SRAI";
    case INST_ADD & 0x00FFFFFF: return "ADD";
    case INST_SUB & 0x00FFFFFF: return "SUB";
    case INST_SLL & 0x00FFFFFF: return "SLL";
    case INST_SLT & 0x00FFFFFF: return "SLT";
    case INST_SLTU & 0x00FFFFFF: return "SLTU";
    case INST_XOR & 0x00FFFFFF: return "XOR";
    case INST_SRL & 0x00FFFFFF: return "SRL";
    case INST_SRA & 0x00FFFFFF: return "SRA";
    case INST_OR & 0x00FFFFFF: return "OR";
    case INST_AND & 0x00FFFFFF: return "AND";
    case INST_MUL & 0x00FFFFFF: return "MUL";
    case INST_MULH & 0x00FFFFFF: return "MULH";
    case INST_MULHSU & 0x00FFFFFF: return "MULHSU";
    case INST_MULHU & 0x00FFFFFF: return "MULHU";
    case INST_DIV & 0x00FFFFFF: return "DIV";
    case INST_DIVU & 0x00FFFFFF: return "DIVU";
    case INST_REM & 0x00FFFFFF: return "REM";
    case INST_REMU & 0x00FFFFFF: return "REMU";
    case INST_EBREAK & 0x00FFFFFF: return "EBREAK";
    case INST_FADD_S & 0x00FFFFFF: return "FADD.S";
    case INST_FSUB_S & 0x00FFFFFF: return "FSUB.S";
    case INST_FMUL_S & 0x00FFFFFF: return "FMUL.S";
    case INST_FDIV_S & 0x00FFFFFF: return "FDIV.S";
    case INST_FLW & 0x00FFFFFF: return "FLW";
    case INST_FSW & 0x00FFFFFF: return "FSW";
    default: return "?";
    }
}

static const char *event_name(uint32_t code)
{
    switch (code & 0x00FFFFFF) {
    case EVENT_ERROR_EVENT & 0x00FFFFFF: return "ERROR";
    case EVENT_KERNEL_DISPATCH & 0x00FFFFFF: return "KERNEL_DISPATCH";
    case EVENT_KERNEL_COMPLETE & 0x00FFFFFF: return "KERNEL_COMPLETE";
    case EVENT_REG_WRITE & 0x00FFFFFF: return "REG_WRITE";
    case EVENT_REG_READ & 0x00FFFFFF: return "REG_READ";
    case EVENT_STATE_CHANGE & 0x00FFFFFF: return "STATE_CHANGE";
    default: return "EVENT";
    }
}

void gpgpu_inst_trace_set_ring(void *ring)
{
    (void)ring;
}
void gpgpu_event_set_ring(void *ring)
{
    (void)ring;
}

void gpgpu_inst_trace_bin(uint32_t inst_code, ...)
{
    uint32_t nargs = (inst_code >> 24) & 0xF;
    const char *name = inst_name(inst_code);

    fprintf(stderr, "  INST: %s", name);

    va_list args;
    va_start(args, inst_code);
    for (uint32_t i = 0; i < nargs; i++) {
        uint32_t v = va_arg(args, uint32_t);
        fprintf(stderr, " 0x%x", v);
    }
    va_end(args);
    fprintf(stderr, "\n");
}

void vpu_event_write(uint32_t event_code, ...)
{
    uint32_t nargs = (event_code >> 24) & 0xF;
    const char *name = event_name(event_code);

    fprintf(stderr, "  EVENT %s:", name);

    va_list args;
    va_start(args, event_code);
    for (uint32_t i = 0; i < nargs; i++) {
        uint32_t v = va_arg(args, uint32_t);
        fprintf(stderr, " 0x%x(%u)", v, v);
    }
    va_end(args);
    fprintf(stderr, "\n");
}
