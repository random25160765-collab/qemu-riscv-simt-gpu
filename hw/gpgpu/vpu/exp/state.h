/*
 * GPGPU State Header — standalone (zero external dependencies)
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * Simplified for standalone interpreter performance testing.
 */

#ifndef GPGPU_STATE_H
#define GPGPU_STATE_H

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

/*
 * ============================================================================
 * 错误状态位掩码 (error_status 寄存器)
 * ============================================================================
 */
#define GPGPU_ERR_INVALID_CMD (1 << 0)
#define GPGPU_ERR_VRAM_FAULT (1 << 1)
#define GPGPU_ERR_KERNEL_FAULT (1 << 2)
#define GPGPU_ERR_DMA_FAULT (1 << 3)

/*
 * ============================================================================
 * 内核分发参数
 * ============================================================================
 */
typedef struct GPGPUKernelParams {
    uint64_t kernel_addr;
    uint64_t kernel_args;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t shared_mem_size;
} GPGPUKernelParams;

/*
 * ============================================================================
 * DMA 状态
 * ============================================================================
 */
typedef struct GPGPUDMAState {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t ctrl;
    uint32_t status;
} GPGPUDMAState;

/*
 * ============================================================================
 * SIMT 执行上下文
 * ============================================================================
 */
typedef struct GPGPUSIMTContext {
    uint32_t thread_id[3];
    uint32_t block_id[3];
    uint32_t warp_id;
    uint32_t lane_id;

    uint32_t barrier_count;
    uint32_t barrier_target;
    bool barrier_active;

    uint32_t thread_mask;
} GPGPUSIMTContext;

/*
 * ============================================================================
 * GPGPU 进程内状态 (纯 C11, 零外部依赖)
 * ============================================================================
 */
typedef struct GPGPUState {
    /* 设备配置 */
    vpu_config_t cfg;
    uint32_t warp_size;
    uint64_t vram_size;

    /* VRAM (本地堆分配) */
    uint8_t *vram_ptr;

    /* Shared memory (per-block, 运行时分配) */
    uint8_t *shm_ptr;
    uint32_t shm_size;

    /* 全局控制寄存器 */
    uint32_t global_ctrl;
    uint32_t global_status;
    uint32_t error_status;

    /* 中断状态 */
    uint32_t irq_enable;
    uint32_t irq_status;

    /* 内核分发参数 */
    GPGPUKernelParams kernel;
    uint32_t kern_size; /* 实际 kernel 大小 (bytes) */

    /* DMA 引擎状态 */
    GPGPUDMAState dma;

    /* SIMT 执行上下文 */
    GPGPUSIMTContext simt;

    /* TCU MMA 状态 */
    struct {
        uint32_t M, K, N; /* 矩阵维度 */
        uint32_t a_base;  /* A 矩阵 VRAM 基址 */
        uint32_t b_base;  /* B 矩阵 VRAM 基址 */
        uint32_t c_base;  /* C 矩阵 VRAM 基址 */
        uint32_t fmt_in;  /* 输入精度: 0=fp32,1=fp16,2=bf16,3=int8 */
        uint32_t fmt_out; /* 输出精度 */
    } mma;

    /* 性能统计 (per-kernel-launch, scheduler 清零) */
    struct {
        uint64_t total_warps;    /* 总 warp 数 (scheduler) */
        uint64_t kernel_ops;     /* 预译码指令条数 (scheduler) */
        uint64_t cat[10];        /* 指令分类: 动态 (engine NEXT, perf=1) */
        uint64_t cat_static[10]; /* 指令分类: 静态 (scheduler 离线) */
        uint64_t total_branches; /* 总分支数 (engine DIV_BR) */
        uint64_t simt_diverges;  /* 分歧分支数 (engine DIV_BR) */
        uint64_t bytes_read;     /* VRAM 读 (engine load handlers) */
        uint64_t bytes_write;    /* VRAM 写 (engine store handlers) */
        uint64_t cache_hits;     /* sector cache 命中 (memory.c) */
        uint64_t cache_misses;   /* sector cache 缺失 (memory.c) */
        uint64_t coal_ops;       /* 合并访问: 32-lane 在同一 cache line (engine loads) */
        uint64_t coal_total;     /* 总 load 操作数 */
    } stats;
} GPGPUState;

#endif /* GPGPU_STATE_H */
