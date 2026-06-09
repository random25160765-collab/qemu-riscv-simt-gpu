/*
 * rvv.h — RVV 模块入口
 *
 * engine.c 通过 #include "rvv.h" 引入所有 RVV handler。
 * 编译时需将 rvv/ 目录加入 include path (-I exp/rvv)。
 *
 * 架构:
 *   rvv_spec.yaml  →  gen_rvv.py  →  generated/rvv_*.h
 *   声明式规格         代码生成器        handler 实现
 */

#include "handles/rvv_all.h"
