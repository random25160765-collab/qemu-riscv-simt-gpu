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

/* Shared register/bitfield/protocol constants (single source of truth) */
#include "vpu/iface.h"

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
 * MSI-X 配置
 * ============================================================================
 */
#define GPGPU_MSIX_VECTORS          4
#define GPGPU_MSIX_VEC_KERNEL       0
#define GPGPU_MSIX_VEC_DMA          1
#define GPGPU_MSIX_VEC_ERROR        2

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
    int error_fd;

    /* VPU 子进程 */
    pid_t vpu_pid;
    bool vpu_crashed;

    /* QEMU 侧维护的 GPU 状态 (避免高频 IPC 轮询) */
    uint32_t irq_enable;
    uint32_t irq_status;
    uint32_t global_status;
};

#endif /* HW_GPGPU_H */
