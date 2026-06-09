/*
 * module/modules.h — 引擎模块总入口
 *
 * engine.c 通过 #include "modules.h" 引入所有 handler。
 * 编译时需将 exp/ 目录加入 include path (-I exp)。
 *
 * 架构:
 *   inst/{isa}/scripts/{spec,gen}.py  →  inst/{isa}/handles/    (生成 .h)
 *   module/                           →  按硬件模块聚合
 *   core/engine.c                     →  #include "../module/modules.h"
 *
 * 硬件模块:
 *   ALU  — 整数算术
 *   FPU  — 浮点
 *   SFU  — 超越函数
 *   LSU  — 访存 (load/store/atomics)
 *   VPU  — 向量
 *   TCU  — 张量
 *   Ctrl — 控制流 (branch/jump/CSR/ebreak/barrier/tex)
 */

/* 控制单元: 分支 + CSR + SIMT 出口 (必须先于其他模块, 提供 DIV_BR 宏) */
#include "ctrl.h"

#include "alu.h"
#include "fpu.h"
#include "sfu.h"
#include "lsu.h"
#include "vpu.h"
#include "tcu.h"
#include "misc.h"
