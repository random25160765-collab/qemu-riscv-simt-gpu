/*
 * GPGPU State Header — standalone (zero QEMU dependencies)
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * Defines GPGPUState and all GPU-simulation sub-structures, extracted from
 * hw/gpgpu/gpgpu.h to decouple GPGPU core from QEMU's PCIDevice / MemoryRegion.
 */

#ifndef GPGPU_STATE_H
#define GPGPU_STATE_H

#include <stdint.h>
#include <stdbool.h>

/* Forward declaration for ring buffer */
typedef struct ring_buf ring_buf;

typedef struct GPGPUState GPGPUState;

/*
 * ============================================================================
 * 错误状态位掩码 (error_status 寄存器)
 * ============================================================================
 */
#define GPGPU_ERR_INVALID_CMD       (1 << 0)
#define GPGPU_ERR_VRAM_FAULT        (1 << 1)
#define GPGPU_ERR_KERNEL_FAULT      (1 << 2)
#define GPGPU_ERR_DMA_FAULT         (1 << 3)

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
 * GPGPU 进程内状态 (纯 C11, 零 QEMU 依赖)
 * ============================================================================
 */
typedef struct GPGPUState {
    /* 设备配置 */
    uint32_t num_cus;
    uint32_t warps_per_cu;
    uint32_t warp_size;
    uint64_t vram_size;

    /* VRAM (指向共享内存或本地分配) */
    uint8_t *vram_ptr;

    /* 全局控制寄存器 */
    uint32_t global_ctrl;
    uint32_t global_status;
    uint32_t error_status;

    /* 中断状态 */
    uint32_t irq_enable;
    uint32_t irq_status;

    /* 内核分发参数 */
    GPGPUKernelParams kernel;

    /* DMA 引擎状态 */
    GPGPUDMAState dma;

    /* SIMT 执行上下文 */
    GPGPUSIMTContext simt;

    /* 环形缓冲区 (进程内堆内存) */
    ring_buf *fast_ring;
    ring_buf *slow_ring;
} GPGPUState;

#endif /* GPGPU_STATE_H */
