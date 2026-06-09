/*
 * module/alu.h — ALU 模块 (标量整数算术)
 *
 * rv32i 被 alu/branch/lsu 三个模块分割, 所以不能用 rv32i_all.h
 * (会重复定义 op_xxx label)。rv32m 是完整 ISA 映射到本模块, 用 _all.h。
 */

/* RV32I ALU families */
#include "../inst/rv32i/handles/rv32i_alu_imm.h"
#include "../inst/rv32i/handles/rv32i_slti_imm.h"
#include "../inst/rv32i/handles/rv32i_shift_imm.h"
#include "../inst/rv32i/handles/rv32i_alu_reg.h"
#include "../inst/rv32i/handles/rv32i_shift_reg.h"
#include "../inst/rv32i/handles/rv32i_slt_reg.h"
#include "../inst/rv32i/handles/rv32i_upper_imm.h"

/* RV32M: 完整 ISA → 一个 _all.h */
#include "../inst/rv32m/handles/rv32m_all.h"
