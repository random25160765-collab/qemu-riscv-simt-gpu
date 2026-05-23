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
#include <stdatomic.h>
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

/* Memory barriers for host standalone (no QEMU osdep.h).
 * Protocol:
 *   QEMU: write data → smp_wmb() → write cmd  → eventfd_write(doorbell)
 *   VPU:  read cmd   → smp_rmb() → read data
 *   VPU:  write result → smp_wmb() → write cmd=NOP [→ eventfd_write(complete)]
 *   QEMU: read cmd → if NOP: smp_rmb() → read result */
#define smp_rmb() __atomic_thread_fence(__ATOMIC_ACQUIRE)
#define smp_wmb() __atomic_thread_fence(__ATOMIC_RELEASE)

typedef struct VPUCtrl {
    uint32_t cmd;
    uint32_t data[3];
} VPUCtrl;

static int vpu_reg_read(GPGPUState *s, uint32_t offset, uint32_t *val)
{
    switch (offset) {
    case GPGPU_REG_DEV_ID:        *val = GPGPU_DEV_ID_VALUE; break;
    case GPGPU_REG_DEV_VERSION:   *val = GPGPU_DEV_VERSION_VALUE; break;
    case GPGPU_REG_DEV_CAPS:
        *val = (s->num_cus & 0xFF) |
               ((s->warps_per_cu & 0xFF) << 8) |
               ((s->warp_size & 0xFF) << 16);
        break;
    case GPGPU_REG_VRAM_SIZE_LO:  *val = 0x04000000; break;
    case GPGPU_REG_VRAM_SIZE_HI:  *val = 0x00000000; break;
    case GPGPU_REG_GLOBAL_CTRL:   *val = s->global_ctrl; break;
    case GPGPU_REG_GLOBAL_STATUS: *val = s->global_status; break;
    case GPGPU_REG_ERROR_STATUS:  *val = s->error_status; break;
    case GPGPU_REG_IRQ_ENABLE:    *val = s->irq_enable; break;
    case GPGPU_REG_IRQ_STATUS:    *val = s->irq_status; break;
    case GPGPU_REG_KERNEL_ADDR_LO: *val = (uint32_t)s->kernel.kernel_addr; break;
    case GPGPU_REG_KERNEL_ADDR_HI: *val = (uint32_t)(s->kernel.kernel_addr >> 32); break;
    case GPGPU_REG_KERNEL_ARGS_LO: *val = (uint32_t)s->kernel.kernel_args; break;
    case GPGPU_REG_KERNEL_ARGS_HI: *val = (uint32_t)(s->kernel.kernel_args >> 32); break;
    case GPGPU_REG_SHARED_MEM_SIZE: *val = s->kernel.shared_mem_size; break;
    case GPGPU_REG_GRID_DIM_X:    *val = s->kernel.grid_dim[0]; break;
    case GPGPU_REG_GRID_DIM_Y:    *val = s->kernel.grid_dim[1]; break;
    case GPGPU_REG_GRID_DIM_Z:    *val = s->kernel.grid_dim[2]; break;
    case GPGPU_REG_BLOCK_DIM_X:   *val = s->kernel.block_dim[0]; break;
    case GPGPU_REG_BLOCK_DIM_Y:   *val = s->kernel.block_dim[1]; break;
    case GPGPU_REG_BLOCK_DIM_Z:   *val = s->kernel.block_dim[2]; break;
    case GPGPU_REG_DMA_SRC_LO:    *val = (uint32_t)s->dma.src_addr; break;
    case GPGPU_REG_DMA_SRC_HI:    *val = (uint32_t)(s->dma.src_addr >> 32); break;
    case GPGPU_REG_DMA_DST_LO:    *val = (uint32_t)s->dma.dst_addr; break;
    case GPGPU_REG_DMA_DST_HI:    *val = (uint32_t)(s->dma.dst_addr >> 32); break;
    case GPGPU_REG_DMA_SIZE:      *val = s->dma.size; break;
    case GPGPU_REG_DMA_CTRL:      *val = s->dma.ctrl; break;
    case GPGPU_REG_DMA_STATUS:    *val = s->dma.status; break;
    case GPGPU_REG_THREAD_ID_X:   *val = s->simt.thread_id[0]; break;
    case GPGPU_REG_THREAD_ID_Y:   *val = s->simt.thread_id[1]; break;
    case GPGPU_REG_THREAD_ID_Z:   *val = s->simt.thread_id[2]; break;
    case GPGPU_REG_BLOCK_ID_X:    *val = s->simt.block_id[0]; break;
    case GPGPU_REG_BLOCK_ID_Y:    *val = s->simt.block_id[1]; break;
    case GPGPU_REG_BLOCK_ID_Z:    *val = s->simt.block_id[2]; break;
    case GPGPU_REG_WARP_ID:       *val = s->simt.warp_id; break;
    case GPGPU_REG_LANE_ID:       *val = s->simt.lane_id; break;
    case GPGPU_REG_THREAD_MASK:   *val = s->simt.thread_mask; break;
    default:
        return -1;
    }
    return 0;
}

static int vpu_reg_write(GPGPUState *s, uint32_t offset, uint32_t val)
{
    switch (offset) {
    case GPGPU_REG_GLOBAL_CTRL:
        if (val & GPGPU_CTRL_RESET) {
            s->global_ctrl = 0;
            s->global_status = GPGPU_STATUS_READY;
            s->error_status = 0;
            s->irq_status = 0;
            memset(&s->simt, 0, sizeof(s->simt));
            memset(&s->kernel, 0, sizeof(s->kernel));
            memset(&s->dma, 0, sizeof(s->dma));
        } else {
            s->global_ctrl = val;
        }
        break;
    case GPGPU_REG_ERROR_STATUS: s->error_status &= ~val; break;
    case GPGPU_REG_IRQ_ENABLE:   s->irq_enable = val; break;
    case GPGPU_REG_IRQ_ACK:      s->irq_status &= ~val; break;
    case GPGPU_REG_KERNEL_ADDR_LO: s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case GPGPU_REG_KERNEL_ADDR_HI: s->kernel.kernel_addr = (s->kernel.kernel_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case GPGPU_REG_KERNEL_ARGS_LO: s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) | val; break;
    case GPGPU_REG_KERNEL_ARGS_HI: s->kernel.kernel_args = (s->kernel.kernel_args & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case GPGPU_REG_GRID_DIM_X:   s->kernel.grid_dim[0] = val; break;
    case GPGPU_REG_GRID_DIM_Y:   s->kernel.grid_dim[1] = val; break;
    case GPGPU_REG_GRID_DIM_Z:   s->kernel.grid_dim[2] = val; break;
    case GPGPU_REG_BLOCK_DIM_X:  s->kernel.block_dim[0] = val; break;
    case GPGPU_REG_BLOCK_DIM_Y:  s->kernel.block_dim[1] = val; break;
    case GPGPU_REG_BLOCK_DIM_Z:  s->kernel.block_dim[2] = val; break;
    case GPGPU_REG_SHARED_MEM_SIZE: s->kernel.shared_mem_size = val; break;
    case GPGPU_REG_DMA_SRC_LO:   s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case GPGPU_REG_DMA_SRC_HI:   s->dma.src_addr = (s->dma.src_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case GPGPU_REG_DMA_DST_LO:   s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) | val; break;
    case GPGPU_REG_DMA_DST_HI:   s->dma.dst_addr = (s->dma.dst_addr & 0x00000000FFFFFFFFULL) | ((uint64_t)val << 32); break;
    case GPGPU_REG_DMA_SIZE:     s->dma.size = val; break;
    case GPGPU_REG_DMA_CTRL:
        if (val & GPGPU_DMA_START) {
            bool dir = (val & GPGPU_DMA_DIR_FROM_VRAM) != 0;
            uint64_t src = s->dma.src_addr;
            uint64_t dst = s->dma.dst_addr;
            uint32_t sz = s->dma.size;
            if (dst + sz > s->vram_size || sz > s->vram_size) {
                s->dma.status = GPGPU_DMA_ERROR;
            } else if (dir) {
                if (src + sz <= s->vram_size)
                    memcpy(s->vram_ptr + dst, s->vram_ptr + src, sz);
                else
                    s->dma.status = GPGPU_DMA_ERROR;
            } else {
                s->dma.status = GPGPU_DMA_ERROR;
            }
            if (s->dma.status != GPGPU_DMA_ERROR)
                s->dma.status = GPGPU_DMA_COMPLETE;
        }
        s->dma.ctrl = val;
        break;
    case GPGPU_REG_DMA_STATUS:   break; /* read-only */
    case GPGPU_REG_THREAD_ID_X:  s->simt.thread_id[0] = val; break;
    case GPGPU_REG_THREAD_ID_Y:  s->simt.thread_id[1] = val; break;
    case GPGPU_REG_THREAD_ID_Z:  s->simt.thread_id[2] = val; break;
    case GPGPU_REG_BLOCK_ID_X:   s->simt.block_id[0] = val; break;
    case GPGPU_REG_BLOCK_ID_Y:   s->simt.block_id[1] = val; break;
    case GPGPU_REG_BLOCK_ID_Z:   s->simt.block_id[2] = val; break;
    case GPGPU_REG_WARP_ID:      s->simt.warp_id = val; break;
    case GPGPU_REG_LANE_ID:      s->simt.lane_id = val; break;
    case GPGPU_REG_BARRIER:      break; /* no-op */
    case GPGPU_REG_THREAD_MASK:  s->simt.thread_mask = val; break;
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
    s.vram_size = 64 * 1024 * 1024; /* 64 MB default */

    /* Override VRAM size from environment (set by QEMU) */
    const char *vram_size_str = getenv(VPU_ENV_VRAM_SIZE);
    if (vram_size_str) {
        uint64_t env_size = strtoull(vram_size_str, NULL, 10);
        long page_sz = sysconf(_SC_PAGESIZE);
        if (env_size > 0 && page_sz > 0 && (env_size % (uint64_t)page_sz) == 0) {
            s.vram_size = env_size;
        }
    }

    /* Get eventfd from environment (read early so VPU_FATAL can use error_fd) */
    const char *doorbell_str = getenv(VPU_ENV_DOORBELL_FD);
    const char *complete_str = getenv(VPU_ENV_COMPLETE_FD);
    const char *error_str = getenv(VPU_ENV_ERROR_FD);
    int error_fd = error_str ? atoi(error_str) : -1;

    /* Report fatal error to QEMU (via error_fd) and exit */
    #define VPU_FATAL(msg) do { \
        fprintf(stderr, "VPU fatal: %s", msg); \
        if (errno) fprintf(stderr, ": %s", strerror(errno)); \
        fprintf(stderr, "\n"); \
        if (error_fd >= 0) { uint64_t one = 1; eventfd_write(error_fd, one); } \
    } while (0)

    /* Open VRAM shared memory (created by QEMU) */
    int vram_fd = shm_open(VPU_SHM_VRAM_NAME, O_RDWR, 0600);
    if (vram_fd < 0) { VPU_FATAL("failed to open VRAM shm"); return 1; }
    s.vram_ptr = mmap(NULL, s.vram_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, vram_fd, 0);
    if (s.vram_ptr == MAP_FAILED) { VPU_FATAL("failed to mmap VRAM"); return 1; }
    close(vram_fd);

    /* Open CTRL shared memory (created by QEMU) */
    int ctrl_fd = shm_open(VPU_SHM_CTRL_NAME, O_RDWR, 0600);
    if (ctrl_fd < 0) { VPU_FATAL("failed to open CTRL shm"); return 1; }
    VPUCtrl *ctrl = mmap(NULL, VPU_CTRL_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ctrl_fd, 0);
    if (ctrl == MAP_FAILED) { VPU_FATAL("failed to mmap CTRL"); return 1; }
    close(ctrl_fd);

    if (!doorbell_str || !complete_str) { VPU_FATAL("missing eventfd env vars"); return 1; }
    int doorbell_fd = atoi(doorbell_str);
    int complete_fd = atoi(complete_str);

    /* Init ring buffers */
    s.fast_ring = ring_buf_create(VPU_FAST_RING_SIZE);
    s.slow_ring = ring_buf_create(VPU_SLOW_RING_SIZE);
    if (!s.fast_ring || !s.slow_ring) {
        fprintf(stderr, "VPU: failed to create ring buffers\n");
        return 1;
    }
    gpgpu_inst_trace_set_ring(s.fast_ring);
    gpgpu_event_set_ring(s.slow_ring);

    s.global_status = GPGPU_STATUS_READY;

    /* Main loop */
    while (1) {
        uint64_t val;
        if (eventfd_read(doorbell_fd, &val) < 0) {
            if (errno == EINTR) continue;
            break;
        }

        while (ctrl->cmd != VPU_CMD_NOP) {
            /* Acquire barrier: ensure data[0]/data[1] reads see QEMU's writes */
            smp_rmb();

            /* Only DISPATCH signals completion via eventfd (async IRQ).
             * Sync ops (REG_READ, REG_WRITE, RESET) use cmd==NOP as signal. */
            bool async_irq = false;

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
                /* Write result BEFORE clearing cmd, with release barrier */
                ctrl->data[1] = value;
                smp_wmb();
                GPGPU_EVENT(s.slow_ring, EVENT_REG_READ, offset, value);
                break;
            }
            case VPU_CMD_DISPATCH: {
                s.error_status = 0;
                s.irq_status &= ~GPGPU_IRQ_ERROR;

                if (s.global_status != GPGPU_STATUS_READY ||
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
                    s.global_status = GPGPU_STATUS_BUSY;
                    int ret = gpgpu_core_exec_kernel(&s);
                    s.global_status = GPGPU_STATUS_READY;
                    GPGPU_EVENT(s.slow_ring, EVENT_KERNEL_COMPLETE, ret);
                    ctrl->data[0] = ret;
                }
                smp_wmb();
                async_irq = true;
                break;
            }
            case VPU_CMD_RESET: {
                s.global_ctrl = 0;
                s.global_status = GPGPU_STATUS_READY;
                s.error_status = 0;
                s.irq_enable = 0;
                s.irq_status = 0;
                memset(&s.kernel, 0, sizeof(s.kernel));
                memset(&s.dma, 0, sizeof(s.dma));
                memset(&s.simt, 0, sizeof(s.simt));
                GPGPU_EVENT(s.slow_ring, EVENT_STATE_CHANGE, 0, GPGPU_STATUS_READY);
                break;
            }
            }

            ctrl->cmd = VPU_CMD_NOP;
            if (async_irq) {
                uint64_t one = 1;
                eventfd_write(complete_fd, one);
            }
        }
    }

    /* Cleanup */
    ring_buf_destroy(s.fast_ring);
    ring_buf_destroy(s.slow_ring);
    munmap(s.vram_ptr, s.vram_size);
    munmap(ctrl, VPU_CTRL_SIZE);

    return 0;
}
