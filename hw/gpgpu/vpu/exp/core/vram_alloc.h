/*
 * vram_alloc.h — VRAM Block Allocator (Vortex MemoryAllocator port)
 * Copyright (c) 2024-2025, GPL v2
 *
 * Best-fit page+block allocator with split/coalesce.
 * Reference: ~/courses/vortex/sim/common/mem_alloc.h
 */
#ifndef VRAM_ALLOC_H
#define VRAM_ALLOC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* --- ptr_table slot indices (VRAM[0x80..0xFF]) --- */
#define PTR_SLOT_A 0
#define PTR_SLOT_B 1
#define PTR_SLOT_C 2
#define PTR_SLOT_D 3

#define PTR_TABLE_OFFSET 0x80
#define PTR_TABLE_SIZE 32

/* --- allocator tunables --- */
#define VRAM_BLOCK_ALIGN 64
#define VRAM_PAGE_ALIGN 4096

/* --- data structures --- */

typedef struct VramBlock {
    struct VramBlock *nextFreeS, *prevFreeS; /* size-sorted (descending) */
    struct VramBlock *nextFreeM, *prevFreeM; /* addr-sorted (ascending) */
    struct VramBlock *nextUsed, *prevUsed;
    uint32_t addr;
    uint32_t size;
} VramBlock;

typedef struct VramPage {
    struct VramPage *next;
    uint32_t addr;
    uint32_t size;
    uint32_t blockAlign;
    VramBlock *usedList;
    VramBlock *freeSList; /* sorted by size desc (largest first) */
    VramBlock *freeMList; /* sorted by addr asc */
} VramPage;

typedef struct {
    uint32_t baseAddress;
    uint32_t capacity;
    uint32_t pageAlign;
    uint32_t blockAlign;
    VramPage *pages;
    uint32_t allocated;
} VramAllocator;

struct GPGPUState;

/* --- lifecycle --- */
void vram_alloc_init(struct GPGPUState *s);
void vram_alloc_reset(struct GPGPUState *s);

/* --- core API --- */
uint32_t vram_alloc(struct GPGPUState *s, size_t size);
void vram_free(struct GPGPUState *s, uint32_t addr);
void vram_reserve(struct GPGPUState *s, uint32_t addr, size_t size);

/* --- ptr_table helpers --- */
uint32_t vram_ptr_read(struct GPGPUState *s, int slot);
void vram_ptr_write(struct GPGPUState *s, int slot, uint32_t addr);

#endif /* VRAM_ALLOC_H */
