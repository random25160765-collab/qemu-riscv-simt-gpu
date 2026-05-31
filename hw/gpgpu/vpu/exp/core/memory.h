/*
 * VPU Memory Interface — 简化版 (纯 VRAM 扁平地址空间)
 *
 * 独立测试中不需要 CTRL 寄存器映射，
 * 所有地址直接映射到 VRAM 线性空间。
 */
#ifndef MEMORY_H
#define MEMORY_H

#include <stdint.h>
#include "state.h"

static inline void out_of_bound(GPGPUState *s, uint32_t addr, int len) {
    if (addr + len > s->vram_size) {
        /* OOB: 返回但不中止（容错） */
    }
}

/* Memory IO function */
static inline uint32_t get_read_addr(void *addr, int len) {
    switch (len) {
        case 1:  return *(uint8_t  *)addr;
        case 2:  return *(uint16_t *)addr;
        case 4:  return *(uint32_t *)addr;
        default: return 0;
    }
}

static inline void get_write_addr(void *addr, int len, uint32_t data) {
    switch (len) {
        case 1: *(uint8_t  *)addr = data; return;
        case 2: *(uint16_t *)addr = data; return;
        case 4: *(uint32_t *)addr = data; return;
    }
}

uint32_t gpu_read(GPGPUState *s, uint32_t addr, int len);
void gpu_write(GPGPUState *s, uint32_t addr, int len, uint32_t data);

#endif /* MEMORY_H */
