/*
 * QEMU Educational GPGPU Device — thin PCI frontend
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * QEMU is a thin pipe: PCI BAR exposure + IRQ delivery.
 * GPU simulation state (registers, VRAM, kernel dispatch, RISC-V SIMT)
 * lives in the VPU host process. QEMU and VPU communicate via shared
 * memory + eventfd.
 */

#ifndef HW_GPGPU_H
#define HW_GPGPU_H

#include "hw/pci/pci_device.h"
#include "hw/pci/pci_ids.h"
#include "qom/object.h"

/*
 * ============================================================================
 * 设备类型定义
 * ============================================================================
 */
#define TYPE_GPGPU "gpgpu"
OBJECT_DECLARE_SIMPLE_TYPE(GPGPUState, GPGPU)

/*
 * ============================================================================
 * PCI 配置
 * ============================================================================
 */
#define GPGPU_VENDOR_ID         0x1234
#define GPGPU_DEVICE_ID         0x1337
#define GPGPU_REVISION          0x01
#define GPGPU_CLASS_CODE        PCI_CLASS_DISPLAY_3D

/*
 * ============================================================================
 * BAR 空间大小配置
 * ============================================================================
 */
#define GPGPU_CTRL_BAR_SIZE     (1 * 1024 * 1024)
#define GPGPU_VRAM_BAR_SIZE     (64 * 1024 * 1024)
#define GPGPU_DOORBELL_BAR_SIZE (64 * 1024)

/*
 * ============================================================================
 * 设备默认配置
 * ============================================================================
 */
#define GPGPU_DEFAULT_NUM_CUS       4
#define GPGPU_DEFAULT_WARPS_PER_CU  4
#define GPGPU_DEFAULT_WARP_SIZE     32
#define GPGPU_DEFAULT_VRAM_SIZE     (64 * 1024 * 1024)

/*
 * ============================================================================
 * 控制寄存器偏移量定义 (BAR 0)
 * ============================================================================
 */
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
#define GPGPU_REG_LOG_LEVEL         0x0500
#define GPGPU_REG_DMA_SRC_LO        0x0400
#define GPGPU_REG_DMA_SRC_HI        0x0404
#define GPGPU_REG_DMA_DST_LO        0x0408
#define GPGPU_REG_DMA_DST_HI        0x040C
#define GPGPU_REG_DMA_SIZE          0x0410
#define GPGPU_REG_DMA_CTRL          0x0414
#define GPGPU_REG_DMA_STATUS        0x0418
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

/*
 * ============================================================================
 * 寄存器位域定义
 * ============================================================================
 */
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

/*
 * ============================================================================
 * MSI-X 配置
 * ============================================================================
 */
#define GPGPU_MSIX_VECTORS          4
#define GPGPU_MSIX_VEC_KERNEL       0
#define GPGPU_MSIX_VEC_DMA          1
#define GPGPU_MSIX_VEC_ERROR        2

/*
 * ============================================================================
 * QEMU-VPU IPC 协议常量 (与 vpu/iface.h 保持同步)
 * ============================================================================
 */
#define VPU_SHM_VRAM_NAME   "/vpu_vram"
#define VPU_SHM_CTRL_NAME   "/vpu_ctrl"
#define VPU_ENV_DOORBELL_FD  "VPU_DOORBELL_FD"
#define VPU_ENV_COMPLETE_FD  "VPU_COMPLETE_FD"
#define VPU_CMD_NOP          0
#define VPU_CMD_REG_WRITE    1
#define VPU_CMD_REG_READ     2
#define VPU_CMD_DISPATCH     3
#define VPU_CMD_RESET        4
#define VPU_CTRL_CMD_OFFSET   0
#define VPU_CTRL_DATA_OFFSET  4
#define VPU_CTRL_DATA1_OFFSET 8
#define VPU_CTRL_SIZE         16

/*
 * ============================================================================
 * 设备主状态结构 (QEMU thin frontend — 仅 PCI + IPC 状态)
 * ============================================================================
 */
struct GPGPUState {
    PCIDevice parent_obj;

    /* BAR 内存区域 */
    MemoryRegion ctrl_mmio;
    MemoryRegion vram;
    MemoryRegion doorbell_mmio;

    /* 设备配置 */
    uint32_t num_cus;
    uint32_t warps_per_cu;
    uint32_t warp_size;
    uint64_t vram_size;

    /* VRAM 共享内存 */
    int vram_shm_fd;
    uint8_t *vram_ptr;

    /* CTRL 共享内存 (命令通道) */
    int ctrl_shm_fd;
    uint32_t *ctrl_ptr;

    /* eventfd */
    int doorbell_fd;
    int complete_fd;

    /* VPU 子进程 */
    pid_t vpu_pid;

    /* QEMU 侧维护的 GPU 状态 (避免高频 IPC 轮询) */
    uint32_t irq_enable;
    uint32_t irq_status;
    uint32_t global_status;
};

#endif /* HW_GPGPU_H */
