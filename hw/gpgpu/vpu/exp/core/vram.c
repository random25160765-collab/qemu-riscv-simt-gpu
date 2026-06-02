/*
 * VPU Memory Interface — 含 sector cache 统计
 */
#include <stdint.h>
#include <string.h>
#include "state.h"
#include "gpgpu_core.h"
#include "vram.h"

/* Cache arrays: defined here, declared extern in vram.h for inline access */
#undef CACHE_SECTORS
#undef SECTOR_SHIFT
#define CACHE_SECTORS 128
#define SECTOR_SHIFT 6
uint32_t cache_tags[CACHE_SECTORS];
int cache_valid[CACHE_SECTORS];

void cache_reset(void)
{
    memset(cache_valid, 0, sizeof(cache_valid));
}

uint32_t gpu_read(GPGPUState *s, uint32_t addr, int len)
{
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
        /* perf counters (read-only) */
        case 0x200: return (uint32_t)s->stats.total_warps;
        case 0x204: return (uint32_t)s->stats.kernel_ops;
        case 0x208: return (uint32_t)s->stats.bytes_read;
        case 0x20C: return (uint32_t)s->stats.bytes_write;
        case 0x210: return (uint32_t)s->stats.total_branches;
        case 0x214: return (uint32_t)s->stats.simt_diverges;
        case 0x218: return (uint32_t)s->stats.cat[0];
        case 0x21C: return (uint32_t)s->stats.cat[1];
        case 0x220: return (uint32_t)s->stats.cat[2];
        case 0x224: return (uint32_t)s->stats.cat[3];
        default: return 0;
        }
    }

    out_of_bound(s, addr, len);
    CACHE_ACCESS(s, addr);
    switch (len) {
    case 1: return *(uint8_t *)(s->vram_ptr + addr);
    case 2: return *(uint16_t *)(s->vram_ptr + addr);
    case 4: return *(uint32_t *)(s->vram_ptr + addr);
    default: return 0;
    }
}

void gpu_write(GPGPUState *s, uint32_t addr, int len, uint32_t data)
{
    if (addr >= 0x80001000 && s->shm_ptr) {
        uint32_t off = addr - 0x80001000;
        if (off + len <= s->shm_size) memcpy(s->shm_ptr + off, &data, len);
        return;
    }
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
    CACHE_ACCESS(s, addr);
    switch (len) {
    case 1: *(uint8_t *)(s->vram_ptr + addr) = data; break;
    case 2: *(uint16_t *)(s->vram_ptr + addr) = data; break;
    case 4: *(uint32_t *)(s->vram_ptr + addr) = data; break;
    }
}
