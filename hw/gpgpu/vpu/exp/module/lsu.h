/*
 * module/lsu.h — 访存单元 (Load/Store + Atomics)
 */

/* 标量 load/store */
#include "../inst/rv32i/handles/rv32i_load.h"
#include "../inst/rv32i/handles/rv32i_store.h"

/* 原子操作 (LR/SC + AMO) */
#include "../inst/rv32a/handles/rv32a_all.h"
