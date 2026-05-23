/*
 * VPU-QEMU Interface Header — shared protocol constants
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * Included by both QEMU (gpgpu.c) and VPU (main.c) to agree on:
 *   - Shared memory segment names
 *   - eventfd environment variable keys
 *   - Control channel command codes
 */

#ifndef VPU_IFACE_H
#define VPU_IFACE_H

/* Shared memory names (passed to shm_open) */
#define VPU_SHM_VRAM_NAME   "/vpu_vram"
#define VPU_SHM_CTRL_NAME   "/vpu_ctrl"

/* eventfd passed from QEMU to VPU via environment variables */
#define VPU_ENV_DOORBELL_FD  "VPU_DOORBELL_FD"
#define VPU_ENV_COMPLETE_FD  "VPU_COMPLETE_FD"
#define VPU_ENV_ERROR_FD     "VPU_ERROR_FD"

/* Control channel command codes (written to CTRL shared memory) */
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

/* Doorbell register layout (BAR4) */
#define VPU_DOORBELL_OFFSET   0   /* write any value to ring doorbell */

/* Ring buffer sizes (in VPU process heap) */
#define VPU_FAST_RING_SIZE    (4 * 1024 * 1024)   /* 4MB — inst trace */
#define VPU_SLOW_RING_SIZE    (64 * 1024)          /* 64KB — ctrl events */

#endif /* VPU_IFACE_H */
