/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

/* Forward declarations */
static void gpgpu_kernel_complete(void *opaque);

/* TODO: Implement MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *gpu = opaque;
    uint64_t val = ~0ULL;

    qemu_log("[DEVICE]: Reading control register 0x%lx, size=%u\n", addr, size);

    switch (addr) {
        case GPGPU_REG_DEV_ID:
            val = GPGPU_DEV_ID_VALUE;
            qemu_log("[DEVICE]: Read DEV_ID: 0x%lx\n", val);
            break;
        case GPGPU_REG_DEV_VERSION:
            val = GPGPU_DEV_VERSION_VALUE;
            qemu_log("[DEVICE]: Read DEV_VERSION: 0x%lx\n", val);
            break;
        case GPGPU_REG_DEV_CAPS:
            val = (gpu->num_cus & 0xFF) |
                  ((gpu->warps_per_cu & 0xFF) << 8) |
                  ((gpu->warp_size & 0xFF) << 16);
            qemu_log("[DEVICE]: Read DEV_CAPS: CUs=%u, Warps/CU=%u, WarpSize=%u, value=0x%lx\n", 
                     gpu->num_cus, gpu->warps_per_cu, gpu->warp_size, val);
            break;
        case GPGPU_REG_VRAM_SIZE_LO:
            val = 0x04000000;
            qemu_log("[DEVICE]: Read VRAM_SIZE_LO: 0x%lx\n", val);
            break;
        case GPGPU_REG_VRAM_SIZE_HI:
            val = 0x00000000;
            qemu_log("[DEVICE]: Read VRAM_SIZE_HI: 0x%lx\n", val);
            break;
        case GPGPU_REG_GLOBAL_CTRL:
            val = gpu->global_ctrl;
            qemu_log("[DEVICE]: Read GLOBAL_CTRL: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_GLOBAL_STATUS:
            val = gpu->global_status;
            qemu_log("[DEVICE]: Read GLOBAL_STATUS: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_ERROR_STATUS:
            val = gpu->error_status;
            qemu_log("[DEVICE]: Read ERROR_STATUS: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_IRQ_ENABLE:
            val = gpu->irq_enable;
            qemu_log("[DEVICE]: Read IRQ_ENABLE: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_IRQ_STATUS:
            val = gpu->irq_status;
            qemu_log("[DEVICE]: Read IRQ_STATUS: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_KERNEL_ADDR_LO:
            val = gpu->kernel.kernel_addr;
            qemu_log("[DEVICE]: Read KERNEL_ADDR_LO: 0x%lx\n", val);
            break;
        case GPGPU_REG_KERNEL_ADDR_HI:
            val = gpu->kernel.kernel_addr >> 32;
            qemu_log("[DEVICE]: Read KERNEL_ADDR_HI: 0x%lx\n", val);
            break;
        case GPGPU_REG_KERNEL_ARGS_LO:
            val = gpu->kernel.kernel_args;
            qemu_log("[DEVICE]: Read KERNEL_ARGS_LO: 0x%lx\n", val);
            break;
        case GPGPU_REG_KERNEL_ARGS_HI:
            val = gpu->kernel.kernel_args >> 32;
            qemu_log("[DEVICE]: Read KERNEL_ARGS_HI: 0x%lx\n", val);
            break;
        case GPGPU_REG_SHARED_MEM_SIZE:
            val = gpu->kernel.shared_mem_size;
            qemu_log("[DEVICE]: Read SHARED_MEM_SIZE: %u bytes\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_X:
            val = gpu->kernel.grid_dim[0];
            qemu_log("[DEVICE]: Read GRID_DIM_X: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_Y:
            val = gpu->kernel.grid_dim[1];
            qemu_log("[DEVICE]: Read GRID_DIM_Y: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_Z:
            val = gpu->kernel.grid_dim[2];
            qemu_log("[DEVICE]: Read GRID_DIM_Z: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_X:
            val = gpu->kernel.block_dim[0];
            qemu_log("[DEVICE]: Read BLOCK_DIM_X: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_Y:
            val = gpu->kernel.block_dim[1];
            qemu_log("[DEVICE]: Read BLOCK_DIM_Y: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_Z:
            val = gpu->kernel.block_dim[2];
            qemu_log("[DEVICE]: Read BLOCK_DIM_Z: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_DMA_SRC_LO:
            val = (gpu->dma.src_addr << 32) >> 32;
            qemu_log("[DEVICE]: Read DMA_SRC_LO: 0x%lx\n", val);
            break;
        case GPGPU_REG_DMA_SRC_HI:
            val = gpu->dma.src_addr >> 32;
            qemu_log("[DEVICE]: Read DMA_SRC_HI: 0x%lx\n", val);
            break;
        case GPGPU_REG_DMA_DST_LO:
            val = (gpu->dma.dst_addr << 32) >> 32;
            qemu_log("[DEVICE]: Read DMA_DST_LO: 0x%lx\n", val);
            break;
        case GPGPU_REG_DMA_DST_HI:
            val = gpu->dma.dst_addr >> 32;
            qemu_log("[DEVICE]: Read DMA_DST_HI: 0x%lx\n", val);
            break;
        case GPGPU_REG_DMA_SIZE:
            val = gpu->dma.size;
            qemu_log("[DEVICE]: Read DMA_SIZE: %u bytes\n", (uint32_t)val);
            break;
        case GPGPU_REG_DMA_CTRL:
            val = gpu->dma.ctrl;
            qemu_log("[DEVICE]: Read DMA_CTRL: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_DMA_STATUS:
            val = gpu->dma.status;
            qemu_log("[DEVICE]: Read DMA_STATUS: 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_ID_X:
            val = gpu->simt.thread_id[0];
            qemu_log("[DEVICE]: Read THREAD_ID_X: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_ID_Y:
            val = gpu->simt.thread_id[1];
            qemu_log("[DEVICE]: Read THREAD_ID_Y: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_ID_Z:
            val = gpu->simt.thread_id[2];
            qemu_log("[DEVICE]: Read THREAD_ID_Z: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_X:
            val = gpu->simt.block_id[0];
            qemu_log("[DEVICE]: Read BLOCK_ID_X: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_Y:
            val = gpu->simt.block_id[1];
            qemu_log("[DEVICE]: Read BLOCK_ID_Y: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_Z:
            val = gpu->simt.block_id[2];
            qemu_log("[DEVICE]: Read BLOCK_ID_Z: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_WARP_ID:
            val = gpu->simt.warp_id;
            qemu_log("[DEVICE]: Read WARP_ID: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_LANE_ID:
            val = gpu->simt.lane_id;
            qemu_log("[DEVICE]: Read LANE_ID: %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_MASK:
            val = gpu->simt.thread_mask;
            qemu_log("[DEVICE]: Read THREAD_MASK: 0x%lx\n", val);
            break;
        default:
            qemu_log("[DEVICE]: Unknown register read: 0x%lx\n", addr);
    }

    qemu_log("[DEVICE]: Control register read completed: 0x%lx -> 0x%lx\n", addr, val);
    return val;
}

/* TODO: Implement MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *gpu = opaque;

    qemu_log("[DEVICE]: Writing control register 0x%lx, value=0x%lx, size=%u\n", addr, val, size);

    switch (addr) {
        case GPGPU_REG_GLOBAL_CTRL:
            if (val & GPGPU_CTRL_RESET) {
                qemu_log("[DEVICE]: GLOBAL_CTRL: Reset command received\n");
                gpu->global_ctrl = 0;
                gpu->global_status = GPGPU_STATUS_READY;
                gpu->error_status = 0;
                gpu->irq_status = 0;
                memset(&gpu->simt, 0, sizeof(gpu->simt));
                memset(&gpu->kernel, 0, sizeof(gpu->kernel));
                memset(&gpu->dma, 0, sizeof(gpu->dma));
                qemu_log("[DEVICE]: Device reset completed\n");
            } else {
                gpu->global_ctrl = val;
                qemu_log("[DEVICE]: GLOBAL_CTRL set to 0x%x\n", (uint32_t)val);
            }
            break;
        case GPGPU_REG_ERROR_STATUS:
            qemu_log("[DEVICE]: ERROR_STATUS: clearing bits 0x%x\n", (uint32_t)val);
            gpu->error_status &= ~val;
            break;
        case GPGPU_REG_IRQ_ENABLE:
            qemu_log("[DEVICE]: IRQ_ENABLE set to 0x%x\n", (uint32_t)val);
            gpu->irq_enable = val;
            break;
        case GPGPU_REG_IRQ_ACK:
            qemu_log("[DEVICE]: IRQ_ACK: acknowledging interrupts 0x%x\n", (uint32_t)val);
            gpu->irq_status &= ~val;
            break;
        case GPGPU_REG_KERNEL_ADDR_LO:
            gpu->kernel.kernel_addr = (gpu->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) | val;
            qemu_log("[DEVICE]: KERNEL_ADDR_LO set to 0x%lx, full address=0x%lx\n", val, gpu->kernel.kernel_addr);
            break;
        case GPGPU_REG_KERNEL_ADDR_HI:
            gpu->kernel.kernel_addr = (gpu->kernel.kernel_addr & 0x00000000FFFFFFFFULL) | (val << 32);
            qemu_log("[DEVICE]: KERNEL_ADDR_HI set to 0x%lx, full address=0x%lx\n", val, gpu->kernel.kernel_addr);
            break;
        case GPGPU_REG_KERNEL_ARGS_LO:
            gpu->kernel.kernel_args = (gpu->kernel.kernel_args & 0xFFFFFFFF00000000ULL) | val;
            qemu_log("[DEVICE]: KERNEL_ARGS_LO set to 0x%lx, full args=0x%lx\n", val, gpu->kernel.kernel_args);
            break;
        case GPGPU_REG_KERNEL_ARGS_HI:
            gpu->kernel.kernel_args = (gpu->kernel.kernel_args & 0x00000000FFFFFFFFULL) | (val << 32);
            qemu_log("[DEVICE]: KERNEL_ARGS_HI set to 0x%lx, full args=0x%lx\n", val, gpu->kernel.kernel_args);
            break;
        case GPGPU_REG_SHARED_MEM_SIZE:
            gpu->kernel.shared_mem_size = val;
            qemu_log("[DEVICE]: SHARED_MEM_SIZE set to %u bytes\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_X:
            gpu->kernel.grid_dim[0] = val;
            qemu_log("[DEVICE]: GRID_DIM_X set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_Y:
            gpu->kernel.grid_dim[1] = val;
            qemu_log("[DEVICE]: GRID_DIM_Y set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_GRID_DIM_Z:
            gpu->kernel.grid_dim[2] = val;
            qemu_log("[DEVICE]: GRID_DIM_Z set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_X:
            gpu->kernel.block_dim[0] = val;
            qemu_log("[DEVICE]: BLOCK_DIM_X set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_Y:
            gpu->kernel.block_dim[1] = val;
            qemu_log("[DEVICE]: BLOCK_DIM_Y set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_DIM_Z:
            gpu->kernel.block_dim[2] = val;
            qemu_log("[DEVICE]: BLOCK_DIM_Z set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_DISPATCH:
            qemu_log("[DEVICE]: DISPATCH command received\n");
            /* Clear previous error status when starting new kernel */
            gpu->error_status = 0;
            gpu->irq_status &= ~GPGPU_IRQ_ERROR;
            
            if (gpu->global_status != GPGPU_STATUS_READY ||
                gpu->kernel.grid_dim[0] == 0 || gpu->kernel.grid_dim[1] == 0 ||
                gpu->kernel.grid_dim[2] == 0 ||
                gpu->kernel.block_dim[0] == 0 || gpu->kernel.block_dim[1] == 0 ||
                gpu->kernel.block_dim[2] == 0 ||
                gpu->kernel.kernel_addr >= gpu->vram_size) {
                qemu_log("[DEVICE]: DISPATCH failed - invalid parameters\n");
                gpu->error_status |= GPGPU_ERR_INVALID_CMD;
                break;
            }
            qemu_log("[DEVICE]: Starting kernel execution: addr=0x%lx, grid=(%u,%u,%u), block=(%u,%u,%u)\n",
                     gpu->kernel.kernel_addr, gpu->kernel.grid_dim[0], gpu->kernel.grid_dim[1], gpu->kernel.grid_dim[2],
                     gpu->kernel.block_dim[0], gpu->kernel.block_dim[1], gpu->kernel.block_dim[2]);
            gpu->global_status = GPGPU_STATUS_BUSY;
            /* 同步执行kernel，直接调用完成处理函数 */
            gpgpu_kernel_complete(gpu);
            break;
        case GPGPU_REG_DMA_SRC_LO:
            gpu->dma.src_addr = (gpu->dma.src_addr & 0xFFFFFFFF00000000ULL) | val;
            qemu_log("[DEVICE]: DMA_SRC_LO set to 0x%lx, full address=0x%lx\n", val, gpu->dma.src_addr);
            break;
        case GPGPU_REG_DMA_SRC_HI:
            gpu->dma.src_addr = (gpu->dma.src_addr & 0x00000000FFFFFFFFULL) | (val << 32);
            qemu_log("[DEVICE]: DMA_SRC_HI set to 0x%lx, full address=0x%lx\n", val, gpu->dma.src_addr);
            break;
        case GPGPU_REG_DMA_DST_LO:
            gpu->dma.dst_addr = (gpu->dma.dst_addr & 0xFFFFFFFF00000000ULL) | val;
            qemu_log("[DEVICE]: DMA_DST_LO set to 0x%lx, full address=0x%lx\n", val, gpu->dma.dst_addr);
            break;
        case GPGPU_REG_DMA_DST_HI:
            gpu->dma.dst_addr = (gpu->dma.dst_addr & 0x00000000FFFFFFFFULL) | (val << 32);
            qemu_log("[DEVICE]: DMA_DST_HI set to 0x%lx, full address=0x%lx\n", val, gpu->dma.dst_addr);
            break;
        case GPGPU_REG_DMA_SIZE:
            gpu->dma.size = val;
            qemu_log("[DEVICE]: DMA_SIZE set to %u bytes\n", (uint32_t)val);
            break;
        case GPGPU_REG_DMA_CTRL:
            if (val & GPGPU_DMA_START) {
                qemu_log("[DEVICE]: DMA_CTRL: Starting DMA transfer\n");
                bool dir = (val & GPGPU_DMA_DIR_FROM_VRAM) != 0;
                uint64_t src = gpu->dma.src_addr;
                uint64_t dst = gpu->dma.dst_addr;
                uint32_t _size = gpu->dma.size;
                qemu_log("[DEVICE]: DMA: direction=%s, src=0x%lx, dst=0x%lx, size=%u bytes\n",
                         dir ? "FROM_VRAM" : "TO_VRAM", src, dst, _size);

                if (dir) {
                    memcpy(gpu->vram_ptr + dst, gpu->vram_ptr + src, _size);
                } else {
                    memcpy(gpu->vram_ptr + dst, (void *)(uintptr_t)src, _size);
                }

                gpu->dma.ctrl |= GPGPU_DMA_BUSY;
                gpu->dma.status = GPGPU_DMA_BUSY;
                timer_mod_ns(gpu->dma_timer,
                             qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + 1000 * 1000);
                qemu_log("[DEVICE]: DMA transfer started, timer scheduled\n");
            }
            gpu->dma.ctrl = val;
            qemu_log("[DEVICE]: DMA_CTRL set to 0x%x\n", (uint32_t)val);
            break;
        case GPGPU_REG_DMA_STATUS:
            qemu_log("[DEVICE]: DMA_STATUS write ignored (read-only)\n");
            break;
        case GPGPU_REG_THREAD_ID_X:
            gpu->simt.thread_id[0] = val;
            qemu_log("[DEVICE]: THREAD_ID_X set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_ID_Y:
            gpu->simt.thread_id[1] = val;
            qemu_log("[DEVICE]: THREAD_ID_Y set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_THREAD_ID_Z:
            gpu->simt.thread_id[2] = val;
            qemu_log("[DEVICE]: THREAD_ID_Z set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_X:
            gpu->simt.block_id[0] = val;
            qemu_log("[DEVICE]: BLOCK_ID_X set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_Y:
            gpu->simt.block_id[1] = val;
            qemu_log("[DEVICE]: BLOCK_ID_Y set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BLOCK_ID_Z:
            gpu->simt.block_id[2] = val;
            qemu_log("[DEVICE]: BLOCK_ID_Z set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_WARP_ID:
            gpu->simt.warp_id = val;
            qemu_log("[DEVICE]: WARP_ID set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_LANE_ID:
            gpu->simt.lane_id = val;
            qemu_log("[DEVICE]: LANE_ID set to %u\n", (uint32_t)val);
            break;
        case GPGPU_REG_BARRIER:
            qemu_log("[DEVICE]: BARRIER command received\n");
            break;
        case GPGPU_REG_THREAD_MASK:
            gpu->simt.thread_mask = val;
            qemu_log("[DEVICE]: THREAD_MASK set to 0x%lx\n", val);
            break;
        default:
            qemu_log("[DEVICE]: Unknown register write: 0x%lx, value=0x%lx\n", addr, val);
    }

    qemu_log("[DEVICE]: Control register write completed: 0x%lx = 0x%lx\n", addr, val);
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement VRAM read */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *gpu = opaque;
    uint64_t val = ~0ULL;

    qemu_log("[DEVICE]: Reading VRAM at address 0x%lx, size=%u\n", addr, size);

    if(addr + size <= gpu->vram_size) {
        switch (size) {
            case 1:
                val = *(uint8_t*)(gpu->vram_ptr + addr);
                // qemu_log("[DEVICE]: VRAM read byte: 0x%lx -> 0x%02x\n", addr, (uint8_t)val);
                break;
            case 2:
                val = *(uint16_t*)(gpu->vram_ptr + addr);
                // qemu_log("[DEVICE]: VRAM read word: 0x%lx -> 0x%04x\n", addr, (uint16_t)val);
                break;
            case 4:
                val = *(uint32_t*)(gpu->vram_ptr + addr);
                // qemu_log("[DEVICE]: VRAM read dword: 0x%lx -> 0x%08x\n", addr, (uint32_t)val);
                break;
            case 8:
                val = *(uint64_t*)(gpu->vram_ptr + addr);
                // qemu_log("[DEVICE]: VRAM read qword: 0x%lx -> 0x%016lx\n", addr, val);
                break;
        }
    } else {
        qemu_log("[DEVICE]: VRAM read failed - address out of bounds: 0x%lx + %u > 0x%lx\n", 
                 addr, size, gpu->vram_size);
    }

    qemu_log("[DEVICE]: VRAM read completed: 0x%lx -> 0x%lx\n", addr, val);
    return val;
}

/* TODO: Implement VRAM write */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *gpu = opaque;

    qemu_log("[DEVICE]: Writing VRAM at address 0x%lx, value=0x%lx, size=%u\n", addr, val, size);

    if(addr + size <= gpu->vram_size) {
        switch (size) {
            case 1:
                *(uint8_t*)(gpu->vram_ptr + addr) = val;
                // qemu_log("[DEVICE]: VRAM write byte: 0x%lx <- 0x%02x\n", addr, (uint8_t)val);
                break;
            case 2:
                *(uint16_t*)(gpu->vram_ptr + addr) = val;
                // qemu_log("[DEVICE]: VRAM write word: 0x%lx <- 0x%04x\n", addr, (uint16_t)val);
                break;
            case 4:
                *(uint32_t*)(gpu->vram_ptr + addr) = val;
                // qemu_log("[DEVICE]: VRAM write dword: 0x%lx <- 0x%08x\n", addr, (uint32_t)val);
                break;
            case 8:
                *(uint64_t*)(gpu->vram_ptr + addr) = val;
                // qemu_log("[DEVICE]: VRAM write qword: 0x%016lx -> 0x%016lx\n", addr, val);
                break;
        }
    } else {
        qemu_log("[DEVICE]: VRAM write failed - address out of bounds: 0x%lx + %u > 0x%lx\n", 
                 addr, size, gpu->vram_size);
    }

    qemu_log("[DEVICE]: VRAM write completed: 0x%lx = 0x%lx\n", addr, val);
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    GPGPUState *s = GPGPU(opaque);

    qemu_log("[DEVICE]: DMA completion handler called\n");

    s->dma.ctrl &= ~GPGPU_DMA_BUSY;
    s->dma.status = GPGPU_DMA_COMPLETE;
    qemu_log("[DEVICE]: DMA transfer completed, status=0x%x\n", s->dma.status);

    if (s->dma.ctrl & GPGPU_DMA_IRQ_ENABLE) {
        qemu_log("[DEVICE]: DMA IRQ enabled, raising DMA_DONE interrupt\n");
        s->irq_status |= GPGPU_IRQ_DMA_DONE;
        if (s->irq_enable & GPGPU_IRQ_DMA_DONE) {
            if (msix_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI-X for DMA interrupt\n");
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_DMA);
            } else if (msi_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI for DMA interrupt\n");
                msi_notify(&s->parent_obj, GPGPU_MSIX_VEC_DMA);
            } else {
                qemu_log("[DEVICE]: Using legacy INTx for DMA interrupt\n");
                pci_set_irq(&s->parent_obj, 1);
            }
        } else {
            qemu_log("[DEVICE]: DMA_DONE interrupt disabled in IRQ_ENABLE\n");
        }
    } else {
        qemu_log("[DEVICE]: DMA IRQ disabled in DMA_CTRL\n");
    }

    qemu_log("[DEVICE]: DMA completion handler finished\n");
}

/* TODO: Implement kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    GPGPUState *s = GPGPU(opaque);

    qemu_log("[DEVICE]: Kernel completion handler called\n");
    qemu_log("[DEVICE]: Executing kernel at address 0x%lx\n", s->kernel.kernel_addr);

    int ret = gpgpu_core_exec_kernel(s);

    if (ret == 0) {
        qemu_log("[DEVICE]: Kernel execution successful, return code=%d\n", ret);
        s->global_status = GPGPU_STATUS_READY;
        s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
        qemu_log("[DEVICE]: Setting global status to READY, raising KERNEL_DONE interrupt\n");
        
        if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
            qemu_log("[DEVICE]: KERNEL_DONE interrupt enabled, sending interrupt\n");
            if (msix_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI-X for kernel completion interrupt\n");
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_KERNEL);
            } else if (msi_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI for kernel completion interrupt\n");
                msi_notify(&s->parent_obj, GPGPU_MSIX_VEC_KERNEL);
            } else {
                qemu_log("[DEVICE]: Using legacy INTx for kernel completion interrupt\n");
                pci_set_irq(&s->parent_obj, 1);
            }
        } else {
            qemu_log("[DEVICE]: KERNEL_DONE interrupt disabled in IRQ_ENABLE\n");
        }
    } else {
        qemu_log("[DEVICE]: Kernel execution failed, return code=%d\n", ret);
        s->global_status = GPGPU_STATUS_ERROR;
        s->error_status |= GPGPU_ERR_KERNEL_FAULT;
        s->irq_status |= GPGPU_IRQ_ERROR;
        qemu_log("[DEVICE]: Setting global status to ERROR, error_status=0x%x\n", s->error_status);
        
        if (s->irq_enable & GPGPU_IRQ_ERROR) {
            qemu_log("[DEVICE]: ERROR interrupt enabled, sending error interrupt\n");
            if (msix_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI-X for error interrupt\n");
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_ERROR);
            } else if (msi_enabled(&s->parent_obj)) {
                qemu_log("[DEVICE]: Using MSI for error interrupt\n");
                msi_notify(&s->parent_obj, GPGPU_MSIX_VEC_ERROR);
            } else {
                qemu_log("[DEVICE]: Using legacy INTx for error interrupt\n");
                pci_set_irq(&s->parent_obj, 1);
            }
        } else {
            qemu_log("[DEVICE]: ERROR interrupt disabled in IRQ_ENABLE\n");
        }
    }

    qemu_log("[DEVICE]: Kernel completion handler finished\n");
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    qemu_log("[DEVICE]: GPGPU device realization started\n");
    qemu_log("[DEVICE]: Device configuration: CUs=%u, Warps/CU=%u, WarpSize=%u, VRAM=%lu MB\n", 
             s->num_cus, s->warps_per_cu, s->warp_size, s->vram_size / (1024 * 1024));

    pci_config_set_interrupt_pin(pci_conf, 1);
    qemu_log("[DEVICE]: PCI interrupt pin configured\n");

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        qemu_log("[DEVICE]: ERROR - Failed to allocate VRAM of size %lu bytes\n", s->vram_size);
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }
    qemu_log("[DEVICE]: VRAM allocated successfully: %lu bytes at %p\n", s->vram_size, s->vram_ptr);

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);
    qemu_log("[DEVICE]: BAR 0 (control registers) initialized: size=0x%x\n", GPGPU_CTRL_BAR_SIZE);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);
    qemu_log("[DEVICE]: BAR 2 (VRAM) initialized: size=0x%lx\n", s->vram_size);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);
    qemu_log("[DEVICE]: BAR 4 (doorbell registers) initialized: size=0x%x\n", GPGPU_DOORBELL_BAR_SIZE);

    // 尝试初始化 MSI-X
    qemu_log("[DEVICE]: Initializing interrupt mechanisms\n");
    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        // MSI-X 初始化失败，尝试 MSI
        qemu_log("[DEVICE]: MSI-X initialization failed, trying MSI\n");
        if (msi_init(pdev, 0, 1, true, false, errp)) {
            // MSI 也失败，使用传统中断
            // 传统中断不需要特殊初始化，PCI 配置已经设置了中断引脚
            qemu_log("[DEVICE]: Both MSI-X and MSI failed, using legacy INTx\n");
            qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: Both MSI-X and MSI failed, using legacy INTx\n");
        } else {
            qemu_log("[DEVICE]: MSI-X failed, using MSI\n");
            qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: MSI-X failed, using MSI\n");
        }
    } else {
        // MSI-X 初始化成功，也初始化 MSI 作为备选
        qemu_log("[DEVICE]: MSI-X initialization successful\n");
        msi_init(pdev, 0, 1, true, false, errp);
        qemu_log("[DEVICE]: MSI also initialized as backup\n");
    }

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);
    qemu_log("[DEVICE]: DMA and kernel timers created\n");

    s->global_status = GPGPU_STATUS_READY;
    qemu_log("[DEVICE]: GPGPU device realization completed successfully\n");
    qemu_log("[DEVICE]: Device status set to READY\n");
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    qemu_log("[DEVICE]: GPGPU device reset started\n");

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    qemu_log("[DEVICE]: Control registers reset: global_ctrl=0, global_status=READY, error_status=0, irq_enable=0, irq_status=0\n");

    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    qemu_log("[DEVICE]: Kernel, DMA, and SIMT structures cleared\n");

    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    qemu_log("[DEVICE]: DMA and kernel timers stopped\n");

    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
        qemu_log("[DEVICE]: VRAM cleared: %lu bytes\n", s->vram_size);
    } else {
        qemu_log("[DEVICE]: VRAM pointer is NULL, skipping clear\n");
    }

    qemu_log("[DEVICE]: GPGPU device reset completed successfully\n");
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)