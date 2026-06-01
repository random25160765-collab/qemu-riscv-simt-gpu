/*
 * VPU Memory Interface — 独立版（含 CTRL 寄存器映射）
 */

#include <stdint.h>
#include <string.h>
#include "state.h"
#include "gpgpu_core.h"
#include "memory.h"

uint32_t gpu_read(GPGPUState *s, uint32_t addr, int len) {
    /* Shared memory (0x80001000+) */
    if (addr >= 0x80001000 && s->shm_ptr) {
        uint32_t off = addr - 0x80001000;
        if (off + len <= s->shm_size) {
            switch (len) {
                case 1: return *(uint8_t *)(s->shm_ptr + off);
                case 2: return *(uint16_t *)(s->shm_ptr + off);
                case 4: return *(uint32_t *)(s->shm_ptr + off);
            }
        }
        return 0;
    }
    /* CTRL 寄存器映射 (0x80000000 base) */
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        switch (addr - GPGPU_CORE_CTRL_BASE) {
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
            default:  return 0;
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
    /* Shared memory (0x80001000+) */
    if (addr >= 0x80001000 && s->shm_ptr) {
        uint32_t off = addr - 0x80001000;
        if (off + len <= s->shm_size)
            memcpy(s->shm_ptr + off, &data, len);
        return;
    }
    /* CTRL 寄存器映射 */
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        switch (addr - GPGPU_CORE_CTRL_BASE) {
            case 0x00: s->simt.thread_id[0] = data; break;
            case 0x04: s->simt.thread_id[1] = data; break;
            case 0x08: s->simt.thread_id[2] = data; break;
            case 0x10: s->simt.block_id[0] = data; break;
            case 0x14: s->simt.block_id[1] = data; break;
            case 0x18: s->simt.block_id[2] = data; break;
            default: break;
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
