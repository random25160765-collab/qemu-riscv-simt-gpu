/*
 * GPGPU - RISC-V SIMT Core Header (standalone)
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 简化的 RV32I 指令解释器，用于 GPU 核心模拟。
 */

#ifndef GPGPU_CORE_H
#define GPGPU_CORE_H

#include <stdint.h>
#include <stdbool.h>

#include "state.h"

/*
 * ============================================================================
 * 常量定义
 * ============================================================================
 */
#define GPGPU_WARP_SIZE 32 /* 每个 warp 的 lane 数量 */
#define GPGPU_NUM_REGS 32  /* RISC-V 通用寄存器数量 */
#define GPGPU_NUM_FREGS 32 /* RISC-V 浮点寄存器数量 f0-f31 */
#define GPGPU_NUM_VREGS 32 /* RISC-V 向量寄存器数量 v0-v31 */

/* 浮点 CSR 地址 */
#define CSR_FFLAGS 0x001
#define CSR_FRM 0x002
#define CSR_FCSR 0x003

/*
 * ============================================================================
 * mhartid CSR 定义
 * ============================================================================
 * 位域布局:
 *   31        13 12     5 4    0
 *   +---------+--------+------+
 *   | block   | warp   | tid  |
 *   | (19bit) | (8bit) | (5b) |
 *   +---------+--------+------+
 */
#define CSR_MHARTID 0xF14

#define MHARTID_THREAD_BITS 5
#define MHARTID_WARP_BITS 8
#define MHARTID_BLOCK_BITS 19
#define MHARTID_THREAD_MASK 0x1F
#define MHARTID_WARP_MASK 0xFF

#define MHARTID_ENCODE(block, warp, thread) (((block) << 13) | ((warp) << 5) | ((thread) & 0x1F))
#define MHARTID_THREAD(id) ((id) & 0x1F)
#define MHARTID_WARP(id) (((id) >> 5) & 0xFF)
#define MHARTID_BLOCK(id) ((id) >> 13)

/*
 * ============================================================================
 * Lane 状态结构
 * ============================================================================
 */

typedef union {
    uint32_t u32;
    int32_t i32;
    float f32;
    struct {
        uint32_t : 16;
        uint16_t bf16;
    };
    struct {
        uint32_t : 24;
        uint8_t e4m3;
    };
    struct {
        uint32_t : 24;
        uint8_t e5m2;
    };
    struct {
        uint32_t : 28;
        uint8_t e2m1;
    };
} GPURegister;

typedef struct GPGPULane {
    GPURegister gpr[GPGPU_NUM_REGS];  /* 通用寄存器 x0-x31 */
    GPURegister fpr[GPGPU_NUM_FREGS]; /* 浮点寄存器 f0-f31 */
    GPURegister vpr[GPGPU_NUM_VREGS]; /* 向量寄存器 v0-v31 */
    uint32_t pc;                      /* 程序计数器 */
    uint32_t mhartid;                 /* 完整 hart ID (block|warp|lane) */
    uint32_t fcsr;                    /* fflags[4:0] | frm[7:5] */
    bool active;                      /* 是否活跃 */
} GPGPULane;

/*
 * ============================================================================
 * Warp 状态结构
 * ============================================================================
 */
typedef struct GPGPUWarp {
    GPGPULane lanes[GPGPU_WARP_SIZE]; /* 32 个 lane */
    uint32_t active_mask;             /* 活跃掩码，每位代表一个 lane */
    uint32_t thread_id_base;          /* 这个 warp 的起始 thread_id */
    uint32_t warp_id;                 /* warp 在 block 内的编号 */
    uint32_t block_id[3];             /* 所属 block 的 ID */
} GPGPUWarp;

/*
 * ============================================================================
 * CTRL 设备地址定义 (GPU 核心视角)
 * ============================================================================
 */
#define GPGPU_CORE_CTRL_BASE 0x80000000 /* CTRL 基地址 */
#define GPGPU_CORE_CTRL_THREAD_ID_X (GPGPU_CORE_CTRL_BASE + 0x00)
#define GPGPU_CORE_CTRL_THREAD_ID_Y (GPGPU_CORE_CTRL_BASE + 0x04)
#define GPGPU_CORE_CTRL_THREAD_ID_Z (GPGPU_CORE_CTRL_BASE + 0x08)
#define GPGPU_CORE_CTRL_BLOCK_ID_X (GPGPU_CORE_CTRL_BASE + 0x10)
#define GPGPU_CORE_CTRL_BLOCK_ID_Y (GPGPU_CORE_CTRL_BASE + 0x14)
#define GPGPU_CORE_CTRL_BLOCK_ID_Z (GPGPU_CORE_CTRL_BASE + 0x18)
#define GPGPU_CORE_CTRL_BLOCK_DIM_X (GPGPU_CORE_CTRL_BASE + 0x20)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x24)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x28)
#define GPGPU_CORE_CTRL_GRID_DIM_X (GPGPU_CORE_CTRL_BASE + 0x30)
#define GPGPU_CORE_CTRL_GRID_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x34)
#define GPGPU_CORE_CTRL_GRID_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x38)

/* Perf counters (read-only, kernel visible via lw) */
#define GPGPU_CORE_PERF_BASE 0x80000200
#define GPGPU_CORE_PERF_TOTAL_WARPS (GPGPU_CORE_PERF_BASE + 0x00)
#define GPGPU_CORE_PERF_KERNEL_OPS (GPGPU_CORE_PERF_BASE + 0x04)
#define GPGPU_CORE_PERF_BYTES_READ (GPGPU_CORE_PERF_BASE + 0x08)
#define GPGPU_CORE_PERF_BYTES_WRITE (GPGPU_CORE_PERF_BASE + 0x0C)
#define GPGPU_CORE_PERF_TOTAL_BRANCHES (GPGPU_CORE_PERF_BASE + 0x10)
#define GPGPU_CORE_PERF_DIVERGES (GPGPU_CORE_PERF_BASE + 0x14)
#define GPGPU_CORE_PERF_CAT_ALU (GPGPU_CORE_PERF_BASE + 0x18)
#define GPGPU_CORE_PERF_CAT_FP (GPGPU_CORE_PERF_BASE + 0x1C)
#define GPGPU_CORE_PERF_CAT_MEM (GPGPU_CORE_PERF_BASE + 0x20)
#define GPGPU_CORE_PERF_CAT_BR (GPGPU_CORE_PERF_BASE + 0x24)

/*
 * ============================================================================
 * GPGPU 错误码 (用于 EVENT_ERROR_EVENT 的 detail 字段)
 * ============================================================================
 */
#define GPGPU_EVT_VRAM_FAULT 0x01
#define GPGPU_EVT_CSR_ACCESS 0x02
#define GPGPU_EVT_ILLEGAL_INST 0x03
#define GPGPU_EVT_WARP_TIMEOUT 0x04
#define GPGPU_EVT_WARP_FAILED 0x05

/*
 * ============================================================================
 * 执行引擎 API — 统一入口
 * ============================================================================
 */

/* 调度器入口 — 执行一次 kernel launch */
int scheduler_run_kernel(GPGPUState *s);

/* (向后兼容: 旧 API 已移除，请使用 scheduler_run_kernel) */

#endif /* GPGPU_CORE_H */