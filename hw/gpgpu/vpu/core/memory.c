/*
 * VPU Memory Interface
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include <stdint.h>
#include "gpgpu.h"
#include "platform/gpgpu_core.h"
#include "memory.h"

uint32_t gpu_read(GPGPUState *s, uint32_t addr, int len) {
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        uint32_t offset = addr - GPGPU_CORE_CTRL_BASE;
        switch (offset) {
            case 0x00: return s->simt.thread_id[0];
            case 0x04: return s->simt.thread_id[1];
            case 0x08: return s->simt.thread_id[2];
            case 0x10: return s->simt.block_id[0];
            case 0x14: return s->simt.block_id[1];
            case 0x18: return s->simt.block_id[2];
            case 0x20: return s->kernel.block_dim[0];
            case 0x24: return s->kernel.block_dim[1];
            case 0x28: return s->kernel.block_dim[2];
            case 0x30: return s->kernel.grid_dim[0];
            case 0x34: return s->kernel.grid_dim[1];
            case 0x38: return s->kernel.grid_dim[2];
            default:
                return 0xFFFFFFFF;
        }
    }
    out_of_bound(s, addr, len);
    switch (len) {
        case 1: return *(uint8_t *)(s->vram_ptr + addr);
        case 2: return *(uint16_t *)(s->vram_ptr + addr);
        case 4: return *(uint32_t *)(s->vram_ptr + addr);
        default: return 0;
    }
}

void gpu_write(GPGPUState *s, uint32_t addr, int len, uint32_t data) {
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        uint32_t offset = addr - GPGPU_CORE_CTRL_BASE;
        switch (offset) {
            case 0x00: s->simt.thread_id[0] = data; break;
            case 0x04: s->simt.thread_id[1] = data; break;
            case 0x08: s->simt.thread_id[2] = data; break;
            case 0x10: s->simt.block_id[0] = data; break;
            case 0x14: s->simt.block_id[1] = data; break;
            case 0x18: s->simt.block_id[2] = data; break;
            default:
                break;
        }
        return;
    }
    out_of_bound(s, addr, len);
    switch (len) {
        case 1: *(uint8_t *)(s->vram_ptr + addr) = data; break;
        case 2: *(uint16_t *)(s->vram_ptr + addr) = data; break;
        case 4: *(uint32_t *)(s->vram_ptr + addr) = data; break;
    }
}