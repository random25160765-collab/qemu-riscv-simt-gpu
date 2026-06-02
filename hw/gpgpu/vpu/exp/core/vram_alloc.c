/*
 * vram_alloc.c — VRAM Bitmap Page Allocator
 * Copyright (c) 2024-2025, GPL v2
 *
 * 4KB-page first-fit bitmap allocator, inspired by Vortex MemoryAllocator.
 * Simple: mark bits in a bitmap, first-fit scan, no split/coalesce needed.
 */
#include "vram_alloc.h"
#include "../state.h"
#include <string.h>

/* --- bitmap helpers --- */
static inline bool bitmap_test(const uint8_t *bm, uint32_t page)
{
    return (bm[page >> 3] >> (page & 7)) & 1;
}
static inline void bitmap_set(uint8_t *bm, uint32_t page)
{
    bm[page >> 3] |= (uint8_t)(1U << (page & 7));
}
static inline void bitmap_clear(uint8_t *bm, uint32_t page)
{
    bm[page >> 3] &= (uint8_t) ~(1U << (page & 7));
}

/* --- lifecycle --- */

/* vram_alloc_init: initial full reset, then reserve fixed regions. */
void vram_alloc_init(struct GPGPUState *s)
{
    /* 1. clear entire bitmap */
    memset(s->vram_bitmap, 0, sizeof(s->vram_bitmap));

    /* 2. reserve param area: 0x000000 - 0x001000 (scalar params + ptr_table) */
    vram_reserve(s, 0, 0x1000);

    /* 3. reserve legacy data region: 0x100000 - 0x500000
     *    (vecmul, gelu, softmax, etc. still hardcode these addresses) */
    vram_reserve(s, 0x100000, 0x400000);

    /* 4. reserve kernel code area: 0x500000 - 0x510000 */
    vram_reserve(s, 0x500000, 0x10000);

    /* 5. zero the ptr_table */
    memset(s->vram_ptr + PTR_TABLE_OFFSET, 0, PTR_TABLE_SIZE);
}

/* vram_alloc_reset: free all user allocations, keep reserved regions.
 * Re-run vram_alloc_init logic. */
void vram_alloc_reset(struct GPGPUState *s)
{
    vram_alloc_init(s);
}

/* --- core API --- */

/* vram_alloc: first-fit scan for 'npages' consecutive free pages.
 * Returns VRAM byte offset, or 0 on OOM. */
uint32_t vram_alloc(struct GPGPUState *s, size_t size)
{
    uint32_t npages = (uint32_t)((size + VRAM_PAGE_SIZE - 1) / VRAM_PAGE_SIZE);
    uint32_t max_page = (uint32_t)(s->vram_size / VRAM_PAGE_SIZE);
    if (npages == 0 || npages > max_page) return 0;

    uint32_t run_start = 0;
    uint32_t run_len = 0;

    for (uint32_t p = 0; p < max_page; p++) {
        if (!bitmap_test(s->vram_bitmap, p)) {
            if (run_len == 0) run_start = p;
            run_len++;
            if (run_len >= npages) {
                /* mark pages as used */
                for (uint32_t i = run_start; i < run_start + npages; i++)
                    bitmap_set(s->vram_bitmap, i);
                return run_start * VRAM_PAGE_SIZE;
            }
        } else {
            run_len = 0;
        }
    }
    return 0; /* OOM */
}

/* vram_free: clear bitmap bits for the given address range.
 * addr must be page-aligned. */
void vram_free(struct GPGPUState *s, uint32_t addr)
{
    uint32_t page = addr / VRAM_PAGE_SIZE;
    uint32_t max_page = (uint32_t)(s->vram_size / VRAM_PAGE_SIZE);

    /* find the extent: clear consecutive used pages starting at 'page' */
    while (page < max_page && bitmap_test(s->vram_bitmap, page))
        bitmap_clear(s->vram_bitmap, page++);
}

/* vram_reserve: mark a fixed address range as used. */
void vram_reserve(struct GPGPUState *s, uint32_t addr, size_t size)
{
    uint32_t start = addr / VRAM_PAGE_SIZE;
    uint32_t npages = (uint32_t)((size + VRAM_PAGE_SIZE - 1) / VRAM_PAGE_SIZE);
    uint32_t max_page = (uint32_t)(s->vram_size / VRAM_PAGE_SIZE);

    for (uint32_t p = start; p < start + npages && p < max_page; p++)
        bitmap_set(s->vram_bitmap, p);
}

/* --- ptr_table helpers --- */
uint32_t vram_ptr_read(struct GPGPUState *s, int slot)
{
    return *(uint32_t *)(s->vram_ptr + PTR_TABLE_OFFSET + (unsigned)slot * 4);
}
void vram_ptr_write(struct GPGPUState *s, int slot, uint32_t addr)
{
    *(uint32_t *)(s->vram_ptr + PTR_TABLE_OFFSET + (unsigned)slot * 4) = addr;
}
