/*
 * vram_alloc.h — VRAM Bitmap Page Allocator
 * Copyright (c) 2024-2025, GPL v2
 *
 * 4KB-page first-fit bitmap allocator.
 * Reference: ~/courses/vortex/sim/common/mem_alloc.h (MemoryAllocator)
 */
#ifndef VRAM_ALLOC_H
#define VRAM_ALLOC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* --- page table slot indices (VRAM[0x80..0xFF]) --- */
#define PTR_SLOT_A 0 /* input buffer A */
#define PTR_SLOT_B 1 /* input buffer B */
#define PTR_SLOT_C 2 /* output buffer C */
#define PTR_SLOT_D 3 /* auxiliary / temp */

#define PTR_TABLE_OFFSET 0x80
#define PTR_TABLE_SIZE 32 /* 8 slots × 4 bytes */

struct GPGPUState;

/* --- lifecycle --- */
void vram_alloc_init(struct GPGPUState *s);  /* reset bitmap, reserve param area */
void vram_alloc_reset(struct GPGPUState *s); /* free all user allocations */

/* --- core API --- */
uint32_t vram_alloc(struct GPGPUState *s, size_t size);              /* first-fit, 0 on OOM */
void vram_free(struct GPGPUState *s, uint32_t addr);                 /* release pages */
void vram_reserve(struct GPGPUState *s, uint32_t addr, size_t size); /* mark used */

/* --- helpers for test harness --- */
uint32_t vram_ptr_read(struct GPGPUState *s, int slot);
void vram_ptr_write(struct GPGPUState *s, int slot, uint32_t addr);

#endif /* VRAM_ALLOC_H */
