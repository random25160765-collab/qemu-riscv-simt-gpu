/*
 * VPU Standalone Process — main entry point
 *
 * Copyright (c) 2024-2025
 *
 * Licensed under GPL v2 or later.
 *
 * Host-side process that owns all GPU simulation state (registers, VRAM,
 * kernel dispatch, RISC-V SIMT interpreter). Communicates with QEMU via
 * shared memory + eventfd.
 *
 * Build: part of libvpu.a, linked into a standalone host binary.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/eventfd.h>
#include <fcntl.h>
#include <errno.h>

#include "state.h"
#include "iface.h"
#include "gpgpu_core.h"
#include "ring/ring.h"
#include "proto/pt_event.h"
#include "core/proto.h"

typedef struct VPUCtrl {
    uint32_t cmd;
    uint32_t data[3];
} VPUCtrl;

static int vpu_reg_read(GPGPUState *s, uint32_t offset, uint32_t *val)
{
    switch (offset) {
    case 0x0000: *val = 0x47505055; break;           /* DEV_ID */
    case 0x0004: *val = 0x00010000; break;            /* DEV_VERSION */
    case 0x0008: /* DEV_CAPS */
        *val = (s->num_cus & 0xFF) |
               ((s->warps_per_cu & 0xFF) << 8) |
               ((s->warp_size & 0xFF) << 16);
        break;
    case 0x000C: *val = 0x04000000; break;            /* VRAM_SIZE_LO */
    case 0x0010: *val = 0x00000000; break;            /* VRAM_SIZE_HI */
    case 0x0100: *val = s->global_ctrl; break;
    case 0x0104: *val = s->global_status; break;
    case 0x0108: *val = s->error_status; break;
    case 0x0200: *val = s->irq_enable; break;
    case 0x0204: *val = s->irq_status; break;
    case 0x0300: *val = (uint32_t)s->kernel.kernel_addr; break;
    case 0x0304: *val = (uint32_t)(s->kernel.kernel_addr >> 32); break;
    case 0x0308: *val = (uint32_t)s->kernel.kernel_args; break;
    case 0x030C: *val = (uint32_t)(s->kernel.kernel_args >> 32); break;
    case 0x0328: *val = s->kernel.shared_mem_size; break;
    case 0x0310: *val = s->kernel.grid_dim[0]; break;
    case 0x0314: *val = s->kernel.grid_dim[1]; break;
    case 0x0318: *val = s->kernel.grid_dim[2]; break;
    case 0x031C: *val = s->kernel.block_dim[0]; break;
    case 0x0320: *val = s->kernel.block_dim[1]; break;
    case 0x0324: *val = s->kernel.block_dim[2]; break;
    case 0x0400: *val = (uint32_t)s->dma.src_addr; break;
    case 0x0404: *val = (uint32_t)(s->dma.src_addr >> 32); break;
    case 0x0408: *val = (uint32_t)s->dma.dst_addr; break;
    case 0x040C: *val = (uint32_t)(s->dma.dst_addr >> 32); break;
    case 0x0410: *val = s->dma.size; break;
    case 0x0414: *val = s->dma.ctrl; break;
    case 0x0418: *val = s->dma.status; break;
    case 0x1000: *val = s->simt.thread_id[0]; break;
    case 0x1004: *val = s->simt.thread_id[1]; break;
    case 0x1008: *val = s->simt.thread_id[2]; break;
    case 0x1010: *val = s->simt.block_id[0]; break;
    case 0x1014: *val = s->simt.block_id[1]; break;
    case 0x1018: *val = s->simt.block_id[2]; break;
    case 0x1020: *val = s->simt.warp_id; break;
    case 0x1024: *val = s->simt.lane_id; break;
    case 0x2004: *val = s->simt.thread_mask; break;
    default:
        return -1;
    }
    return 0;
}

static int vpu_reg_write(GPGPUState *s, uint32_t offset, uint32_t val)
{
    switch (offset) {
    case 0x0100: /* GLOBAL_CTRL */
        if (val & (1 << 1)) { /* reset */
            s->global_ctrl = 0;
            s->global_status = (1 << 0); /* READY */
            s->error_status = 0;
            s->irq_status = 0;
            memset(&s->simt, 0, sizeof(s->simt));
            memset(&s->kernel, 0, sizeof(s->kernel));
            memset(&s->dma, 0, sizeof(s->dma));
        } else {
            s->global_ctrl = val;
        }
        break;
    case 0x0108: s->error_status &= ~val; break;
    case 0x0200: s->irq_enable = val; break;
    case 0x0208: s->irq_status &= ~val; break;
    case 0x0300: s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case 0x0304: s->kernel.kernel_addr = (s->kernel.kernel_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case 0x0308: s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) | val; break;
    case 0x030C: s->kernel.kernel_args = (s->kernel.kernel_args & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case 0x0310: s->kernel.grid_dim[0] = val; break;
    case 0x0314: s->kernel.grid_dim[1] = val; break;
    case 0x0318: s->kernel.grid_dim[2] = val; break;
    case 0x031C: s->kernel.block_dim[0] = val; break;
    case 0x0320: s->kernel.block_dim[1] = val; break;
    case 0x0324: s->kernel.block_dim[2] = val; break;
    case 0x0328: s->kernel.shared_mem_size = val; break;
    case 0x0400: s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case 0x0404: s->dma.src_addr = (s->dma.src_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case 0x0408: s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case 0x040C: s->dma.dst_addr = (s->dma.dst_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case 0x0410: s->dma.size = val; break;
    case 0x0414: /* DMA_CTRL */
        if (val & (1 << 0)) { /* DMA_START */
            bool dir = (val & (1 << 1)) != 0;
            uint64_t src = s->dma.src_addr;
            uint64_t dst = s->dma.dst_addr;
            uint32_t sz = s->dma.size;
            if (dir) {
                memcpy(s->vram_ptr + dst, s->vram_ptr + src, sz);
            } else {
                memcpy(s->vram_ptr + dst, (void *)(uintptr_t)src, sz);
            }
            s->dma.status = (1 << 1); /* DMA_COMPLETE */
        }
        s->dma.ctrl = val;
        break;
    case 0x0418: break; /* DMA_STATUS read-only */
    case 0x1000: s->simt.thread_id[0] = val; break;
    case 0x1004: s->simt.thread_id[1] = val; break;
    case 0x1008: s->simt.thread_id[2] = val; break;
    case 0x1010: s->simt.block_id[0] = val; break;
    case 0x1014: s->simt.block_id[1] = val; break;
    case 0x1018: s->simt.block_id[2] = val; break;
    case 0x1020: s->simt.warp_id = val; break;
    case 0x1024: s->simt.lane_id = val; break;
    case 0x2000: break; /* BARRIER — no-op */
    case 0x2004: s->simt.thread_mask = val; break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    GPGPUState s;
    memset(&s, 0, sizeof(s));

    /* Default device config */
    s.num_cus = 4;
    s.warps_per_cu = 4;
    s.warp_size = 32;
    s.vram_size = 64 * 1024 * 1024; /* 64 MB */

    /* Open VRAM shared memory (created by QEMU) */
    int vram_fd = shm_open(VPU_SHM_VRAM_NAME, O_RDWR, 0600);
    if (vram_fd < 0) {
        fprintf(stderr, "VPU: failed to open VRAM shm: %s\n", strerror(errno));
        return 1;
    }
    s.vram_ptr = mmap(NULL, s.vram_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, vram_fd, 0);
    if (s.vram_ptr == MAP_FAILED) {
        fprintf(stderr, "VPU: failed to mmap VRAM: %s\n", strerror(errno));
        return 1;
    }
    close(vram_fd);

    /* Open CTRL shared memory (created by QEMU) */
    int ctrl_fd = shm_open(VPU_SHM_CTRL_NAME, O_RDWR, 0600);
    if (ctrl_fd < 0) {
        fprintf(stderr, "VPU: failed to open CTRL shm: %s\n", strerror(errno));
        return 1;
    }
    VPUCtrl *ctrl = mmap(NULL, VPU_CTRL_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ctrl_fd, 0);
    if (ctrl == MAP_FAILED) {
        fprintf(stderr, "VPU: failed to mmap CTRL: %s\n", strerror(errno));
        return 1;
    }
    close(ctrl_fd);

    /* Get eventfd from environment */
    const char *doorbell_str = getenv(VPU_ENV_DOORBELL_FD);
    const char *complete_str = getenv(VPU_ENV_COMPLETE_FD);
    if (!doorbell_str || !complete_str) {
        fprintf(stderr, "VPU: missing eventfd env vars\n");
        return 1;
    }
    int doorbell_fd = atoi(doorbell_str);
    int complete_fd = atoi(complete_str);

    /* Init ring buffers */
    s.fast_ring = ring_buf_create(VPU_FAST_RING_SIZE);
    s.slow_ring = ring_buf_create(VPU_SLOW_RING_SIZE);
    if (!s.fast_ring || !s.slow_ring) {
        fprintf(stderr, "VPU: failed to create ring buffers\n");
        return 1;
    }

    s.global_status = 1 << 0; /* READY */

    /* Main loop */
    while (1) {
        uint64_t val;
        if (eventfd_read(doorbell_fd, &val) < 0) {
            if (errno == EINTR) continue;
            break;
        }

        while (ctrl->cmd != VPU_CMD_NOP) {
            switch (ctrl->cmd) {
            case VPU_CMD_REG_WRITE: {
                uint32_t offset = ctrl->data[0];
                uint32_t value = ctrl->data[1];
                vpu_reg_write(&s, offset, value);
                GPGPU_EVENT(s.slow_ring, EVENT_REG_WRITE, offset, value);
                break;
            }
            case VPU_CMD_REG_READ: {
                uint32_t offset = ctrl->data[0];
                uint32_t value;
                vpu_reg_read(&s, offset, &value);
                ctrl->data[1] = value;
                GPGPU_EVENT(s.slow_ring, EVENT_REG_READ, offset, value);
                break;
            }
            case VPU_CMD_DISPATCH: {
                s.error_status = 0;
                s.irq_status &= ~(1 << 2); /* clear ERROR IRQ */

                if (s.global_status != (1 << 0) || /* not READY */
                    s.kernel.grid_dim[0] == 0 ||
                    s.kernel.grid_dim[1] == 0 ||
                    s.kernel.grid_dim[2] == 0 ||
                    s.kernel.block_dim[0] == 0 ||
                    s.kernel.block_dim[1] == 0 ||
                    s.kernel.block_dim[2] == 0 ||
                    s.kernel.kernel_addr >= s.vram_size) {
                    s.error_status |= GPGPU_ERR_INVALID_CMD;
                    ctrl->data[0] = -1;
                } else {
                    GPGPU_EVENT(s.slow_ring, EVENT_KERNEL_DISPATCH,
                                (uint32_t)s.kernel.kernel_addr,
                                s.kernel.grid_dim[0],
                                s.kernel.grid_dim[1],
                                s.kernel.grid_dim[2],
                                s.kernel.block_dim[0],
                                s.kernel.block_dim[1],
                                s.kernel.block_dim[2]);
                    s.global_status = (1 << 1); /* BUSY */
                    int ret = gpgpu_core_exec_kernel(&s);
                    GPGPU_EVENT(s.slow_ring, EVENT_KERNEL_COMPLETE, ret);
                    ctrl->data[0] = ret;
                }
                break;
            }
            case VPU_CMD_RESET: {
                s.global_ctrl = 0;
                s.global_status = 1 << 0; /* READY */
                s.error_status = 0;
                s.irq_enable = 0;
                s.irq_status = 0;
                memset(&s.kernel, 0, sizeof(s.kernel));
                memset(&s.dma, 0, sizeof(s.dma));
                memset(&s.simt, 0, sizeof(s.simt));
                GPGPU_EVENT(s.slow_ring, EVENT_STATE_CHANGE, 0, 1 << 0);
                break;
            }
            }

            ctrl->cmd = VPU_CMD_NOP;
            /* Notify QEMU that command processing is done */
            uint64_t one = 1;
            eventfd_write(complete_fd, one);
        }
    }

    /* Cleanup */
    ring_buf_destroy(s.fast_ring);
    ring_buf_destroy(s.slow_ring);
    munmap(s.vram_ptr, s.vram_size);
    munmap(ctrl, VPU_CTRL_SIZE);

    return 0;
}
