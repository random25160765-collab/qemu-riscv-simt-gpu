/*
 * VPU Memory Interface
 */
#ifndef MEMORY_H
#define MEMORY_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "state.h"

/* debug trace: gpu_config.lua debug=true 开启 */
#define DLOG(s, ...)                                               \
    do {                                                           \
        if ((s)->cfg.features.debug) fprintf(stderr, __VA_ARGS__); \
    } while (0)

static inline void out_of_bound(GPGPUState *s, uint32_t addr, int len)
{
    if (addr + len > s->vram_size) s->error_status |= GPGPU_ERR_VRAM_FAULT;
}

uint32_t gpu_read(GPGPUState *s, uint32_t addr, int len);
void gpu_write(GPGPUState *s, uint32_t addr, int len, uint32_t data);
void cache_reset(void);

/* sector cache — macro 强制 inline, 零函数调用, 数组在 memory.c */
#define CACHE_SECTORS 128
#define SECTOR_SHIFT 6
extern uint32_t cache_tags[CACHE_SECTORS];
extern int cache_valid[CACHE_SECTORS];

#define CACHE_ACCESS(s, addr)                                \
    do {                                                     \
        if (!(s)->cfg.features.perf) break;                  \
        uint32_t _tag = (addr) >> SECTOR_SHIFT;              \
        int _idx = _tag % CACHE_SECTORS;                     \
        if (cache_valid[_idx] && cache_tags[_idx] == _tag) { \
            (s)->stats.cache_hits++;                         \
        } else {                                             \
            (s)->stats.cache_misses++;                       \
            cache_valid[_idx] = 1;                           \
            cache_tags[_idx] = _tag;                         \
        }                                                    \
    } while (0)

#endif /* MEMORY_H */
