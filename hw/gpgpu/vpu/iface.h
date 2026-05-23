/*
 * VPU-QEMU Interface Header — shared protocol constants
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * Included by both QEMU (via gpgpu.h) and VPU (main.c) to agree on:
 *   - Register offset macros (GPGPU_REG_*)
 *   - Register bitfield definitions
 *   - Shared memory segment names
 *   - eventfd environment variable keys
 *   - Control channel command codes
 *
 * SINGLE SOURCE OF TRUTH: both sides use these macros so offsets never
 * drift out of sync.
 */

#ifndef VPU_IFACE_H
#define VPU_IFACE_H

/* =========================================================================
 * Register offset macros (BAR0, BAR4 doorbell)
 * ========================================================================= */
#define GPGPU_REG_DEV_ID            0x0000
#define GPGPU_REG_DEV_VERSION       0x0004
#define GPGPU_REG_DEV_CAPS          0x0008
#define GPGPU_REG_VRAM_SIZE_LO      0x000C
#define GPGPU_REG_VRAM_SIZE_HI      0x0010
#define GPGPU_REG_GLOBAL_CTRL       0x0100
#define GPGPU_REG_GLOBAL_STATUS     0x0104
#define GPGPU_REG_ERROR_STATUS      0x0108
#define GPGPU_REG_IRQ_ENABLE        0x0200
#define GPGPU_REG_IRQ_STATUS        0x0204
#define GPGPU_REG_IRQ_ACK           0x0208
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304
#define GPGPU_REG_KERNEL_ARGS_LO    0x0308
#define GPGPU_REG_KERNEL_ARGS_HI    0x030C
#define GPGPU_REG_GRID_DIM_X        0x0310
#define GPGPU_REG_GRID_DIM_Y        0x0314
#define GPGPU_REG_GRID_DIM_Z        0x0318
#define GPGPU_REG_BLOCK_DIM_X       0x031C
#define GPGPU_REG_BLOCK_DIM_Y       0x0320
#define GPGPU_REG_BLOCK_DIM_Z       0x0324
#define GPGPU_REG_SHARED_MEM_SIZE   0x0328
#define GPGPU_REG_DISPATCH          0x0330
#define GPGPU_REG_DMA_SRC_LO        0x0400
#define GPGPU_REG_DMA_SRC_HI        0x0404
#define GPGPU_REG_DMA_DST_LO        0x0408
#define GPGPU_REG_DMA_DST_HI        0x040C
#define GPGPU_REG_DMA_SIZE          0x0410
#define GPGPU_REG_DMA_CTRL          0x0414
#define GPGPU_REG_DMA_STATUS        0x0418
#define GPGPU_REG_LOG_LEVEL         0x0500
#define GPGPU_REG_THREAD_ID_X       0x1000
#define GPGPU_REG_THREAD_ID_Y       0x1004
#define GPGPU_REG_THREAD_ID_Z       0x1008
#define GPGPU_REG_BLOCK_ID_X        0x1010
#define GPGPU_REG_BLOCK_ID_Y        0x1014
#define GPGPU_REG_BLOCK_ID_Z        0x1018
#define GPGPU_REG_WARP_ID           0x1020
#define GPGPU_REG_LANE_ID           0x1024
#define GPGPU_REG_BARRIER           0x2000
#define GPGPU_REG_THREAD_MASK       0x2004

/* =========================================================================
 * Register bitfield definitions
 * ========================================================================= */
#define GPGPU_CTRL_ENABLE           (1 << 0)
#define GPGPU_CTRL_RESET            (1 << 1)
#define GPGPU_STATUS_READY          (1 << 0)
#define GPGPU_STATUS_BUSY           (1 << 1)
#define GPGPU_STATUS_ERROR          (1 << 2)
#define GPGPU_ERR_INVALID_CMD       (1 << 0)
#define GPGPU_ERR_VRAM_FAULT        (1 << 1)
#define GPGPU_ERR_KERNEL_FAULT      (1 << 2)
#define GPGPU_ERR_DMA_FAULT         (1 << 3)
#define GPGPU_IRQ_KERNEL_DONE       (1 << 0)
#define GPGPU_IRQ_DMA_DONE          (1 << 1)
#define GPGPU_IRQ_ERROR             (1 << 2)
#define GPGPU_DMA_START             (1 << 0)
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)
#define GPGPU_DMA_IRQ_ENABLE        (1 << 2)
#define GPGPU_DMA_BUSY              (1 << 0)
#define GPGPU_DMA_COMPLETE          (1 << 1)
#define GPGPU_DMA_ERROR             (1 << 2)
#define GPGPU_DEV_ID_VALUE          0x47505055
#define GPGPU_DEV_VERSION_VALUE     0x00010000

/* =========================================================================
 * Shared memory names (passed to shm_open)
 * ========================================================================= */
#define VPU_SHM_VRAM_NAME   "/vpu_vram"
#define VPU_SHM_CTRL_NAME   "/vpu_ctrl"

/* =========================================================================
 * eventfd passed from QEMU to VPU via environment variables
 * ========================================================================= */
#define VPU_ENV_DOORBELL_FD  "VPU_DOORBELL_FD"
#define VPU_ENV_COMPLETE_FD  "VPU_COMPLETE_FD"
#define VPU_ENV_ERROR_FD     "VPU_ERROR_FD"
#define VPU_ENV_VRAM_SIZE    "VPU_VRAM_SIZE"

/* =========================================================================
 * Control channel command codes (written to CTRL shared memory)
 * ========================================================================= */
#define VPU_CMD_NOP          0
#define VPU_CMD_REG_WRITE    1   /* data[0]=offset, data[1]=value */
#define VPU_CMD_REG_READ     2   /* data[0]=offset → VPU fills data[1]=value */
#define VPU_CMD_DISPATCH     3   /* launch kernel execution */
#define VPU_CMD_RESET        4   /* soft reset */

/* Control channel layout (one command at a time) */
#define VPU_CTRL_CMD_OFFSET   0   /* command code (uint32_t) */
#define VPU_CTRL_DATA_OFFSET  4   /* data[0] (uint32_t) */
#define VPU_CTRL_DATA1_OFFSET 8   /* data[1] (uint32_t) */
#define VPU_CTRL_SIZE         16  /* total control channel size */

/* =========================================================================
 * Doorbell register layout (BAR4)
 * ========================================================================= */
#define VPU_DOORBELL_OFFSET   0   /* write any value to ring doorbell */

/* =========================================================================
 * Ring buffer sizes (in VPU process heap)
 * ========================================================================= */
#define VPU_FAST_RING_SIZE    (4 * 1024 * 1024)   /* 4MB — inst trace */
#define VPU_SLOW_RING_SIZE    (64 * 1024)          /* 64KB — ctrl events */

#endif /* VPU_IFACE_H */
