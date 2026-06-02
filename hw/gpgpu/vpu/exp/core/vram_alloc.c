/*
 * vram_alloc.c — VRAM Block Allocator (Vortex MemoryAllocator C port)
 * Copyright (c) 2024-2025, GPL v2
 *
 * Best-fit page+block allocator with split on alloc, coalesce on free.
 * Reference: ~/courses/vortex/sim/common/mem_alloc.h
 */
#include "vram_alloc.h"
#include "../state.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================
 *  helpers
 * ============================================================ */

static uint32_t alignSize(uint32_t size, uint32_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1U);
}

static VramBlock *createBlock(uint32_t addr, uint32_t size)
{
    VramBlock *b = calloc(1, sizeof(VramBlock));
    if (b) {
        b->addr = addr;
        b->size = size;
    }
    return b;
}

static void destroyBlock(VramBlock *b)
{
    free(b);
}

/* ============================================================
 *  page-level: list management (insert / remove / find)
 * ============================================================ */

/* --- used list (prepend, unordered) --- */

static void insertUsedList(VramPage *page, VramBlock *block)
{
    block->nextUsed = page->usedList;
    if (page->usedList) page->usedList->prevUsed = block;
    page->usedList = block;
}

static void removeUsedList(VramPage *page, VramBlock *block)
{
    if (block->prevUsed)
        block->prevUsed->nextUsed = block->nextUsed;
    else
        page->usedList = block->nextUsed;
    if (block->nextUsed) block->nextUsed->prevUsed = block->prevUsed;
    block->nextUsed = block->prevUsed = NULL;
}

/* --- free-S list: sorted by SIZE descending (largest first, best-fit scan) --- */

static void insertFreeSList(VramPage *page, VramBlock *block)
{
    VramBlock *curr = page->freeSList, *prev = NULL;
    while (curr && curr->size > block->size) {
        prev = curr;
        curr = curr->nextFreeS;
    }
    block->nextFreeS = curr;
    block->prevFreeS = prev;
    if (prev)
        prev->nextFreeS = block;
    else
        page->freeSList = block;
    if (curr) curr->prevFreeS = block;
}

static void removeFreeSList(VramPage *page, VramBlock *block)
{
    if (block->prevFreeS)
        block->prevFreeS->nextFreeS = block->nextFreeS;
    else
        page->freeSList = block->nextFreeS;
    if (block->nextFreeS) block->nextFreeS->prevFreeS = block->prevFreeS;
    block->nextFreeS = block->prevFreeS = NULL;
}

/* --- free-M list: sorted by ADDRESS ascending (coalesce on free) --- */

static void insertFreeMList(VramPage *page, VramBlock *block)
{
    VramBlock *curr = page->freeMList, *prev = NULL;
    while (curr && curr->addr < block->addr) {
        prev = curr;
        curr = curr->nextFreeM;
    }
    block->nextFreeM = curr;
    block->prevFreeM = prev;
    if (prev)
        prev->nextFreeM = block;
    else
        page->freeMList = block;
    if (curr) curr->prevFreeM = block;
}

static void removeFreeMList(VramPage *page, VramBlock *block)
{
    if (block->prevFreeM)
        block->prevFreeM->nextFreeM = block->nextFreeM;
    else
        page->freeMList = block->nextFreeM;
    if (block->nextFreeM) block->nextFreeM->prevFreeM = block->prevFreeM;
    block->nextFreeM = block->prevFreeM = NULL;
}

/* --- find free block: best-fit --- */

static VramBlock *findFreeBlock(VramPage *page, uint32_t size)
{
    VramBlock *b = page->freeSList;
    if (!b || b->size < size) return NULL;
    /* walk toward smaller blocks, find smallest that fits */
    while (b->nextFreeS && b->nextFreeS->size >= size)
        b = b->nextFreeS;
    return b;
}

/* --- find used block by address --- */

static VramBlock *findUsedBlock(VramPage *page, uint32_t addr)
{
    if (addr < page->addr || addr >= page->addr + page->size) return NULL;
    VramBlock *b = page->usedList;
    while (b) {
        if (b->addr == addr) return b;
        b = b->nextUsed;
    }
    return NULL;
}

/* ============================================================
 *  page-level: allocate / release blocks within a page
 * ============================================================ */

static void pageAllocate(VramPage *page, uint32_t size, VramBlock *freeBlock)
{
    removeFreeMList(page, freeBlock);
    removeFreeSList(page, freeBlock);

    /* split: carve off the excess as a new free block */
    uint32_t extra = freeBlock->size - size;
    if (extra >= page->blockAlign) {
        freeBlock->size = size;
        VramBlock *nb = createBlock(freeBlock->addr + size, extra);
        if (nb) {
            insertFreeMList(page, nb);
            insertFreeSList(page, nb);
        }
    }
    insertUsedList(page, freeBlock);
}

static void pageRelease(VramPage *page, VramBlock *usedBlock)
{
    removeUsedList(page, usedBlock);
    insertFreeMList(page, usedBlock);

    /* coalesce left */
    if (usedBlock->prevFreeM) {
        if (usedBlock->prevFreeM->addr + usedBlock->prevFreeM->size == usedBlock->addr) {
            VramBlock *left = usedBlock->prevFreeM;
            left->size += usedBlock->size;
            left->nextFreeM = usedBlock->nextFreeM;
            if (left->nextFreeM) left->nextFreeM->prevFreeM = left;
            removeFreeSList(page, left); /* size changed, re-insert later */
            destroyBlock(usedBlock);
            usedBlock = left;
        }
    }

    /* coalesce right */
    if (usedBlock->nextFreeM) {
        if (usedBlock->addr + usedBlock->size == usedBlock->nextFreeM->addr) {
            VramBlock *right = usedBlock->nextFreeM;
            usedBlock->size += right->size;
            usedBlock->nextFreeM = right->nextFreeM;
            if (usedBlock->nextFreeM) usedBlock->nextFreeM->prevFreeM = usedBlock;
            removeFreeSList(page, right);
            destroyBlock(right);
        }
    }

    insertFreeSList(page, usedBlock);
}

/* ============================================================
 *  top-level: page list management
 * ============================================================ */

static VramPage *createPage(VramAllocator *a, uint32_t addr, uint32_t size)
{
    VramPage *page = calloc(1, sizeof(VramPage));
    if (!page) return NULL;
    page->addr = addr;
    page->size = size;
    page->blockAlign = a->blockAlign;

    /* initial free block covers entire page */
    VramBlock *fb = createBlock(addr, size);
    if (!fb) {
        free(page);
        return NULL;
    }
    page->freeSList = page->freeMList = fb;

    /* insert into sorted page list */
    if (!a->pages || a->pages->addr > addr) {
        page->next = a->pages;
        a->pages = page;
    } else {
        VramPage *cur = a->pages;
        while (cur->next && cur->next->addr < addr)
            cur = cur->next;
        page->next = cur->next;
        cur->next = page;
    }
    return page;
}

static void destroyPage(VramAllocator *a, VramPage *page)
{
    /* unlink from page list */
    VramPage *prev = NULL, *cur = a->pages;
    while (cur) {
        if (cur == page) {
            if (prev)
                prev->next = cur->next;
            else
                a->pages = cur->next;
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    /* free all blocks (should only have the initial free block left) */
    if (page->freeMList) destroyBlock(page->freeMList);
    free(page);
}

static bool findNextAddress(VramAllocator *a, uint32_t size, uint32_t *addr)
{
    if (!a->pages) {
        *addr = a->baseAddress;
        return true;
    }
    uint32_t endOfLast = a->baseAddress;
    VramPage *cur = a->pages;
    while (cur) {
        if (endOfLast + size <= cur->addr) {
            *addr = endOfLast;
            return true;
        }
        endOfLast = cur->addr + cur->size;
        cur = cur->next;
    }
    if (endOfLast + size <= a->baseAddress + a->capacity) {
        *addr = endOfLast;
        return true;
    }
    return false;
}

static bool hasPageOverlap(VramAllocator *a, uint32_t start, uint32_t size)
{
    uint32_t end = start + size;
    VramPage *cur = a->pages;
    while (cur) {
        if (start < cur->addr + cur->size && end > cur->addr) return true;
        cur = cur->next;
    }
    (void)a;
    return false;
}

/* ============================================================
 *  public API
 * ============================================================ */

void vram_alloc_init(struct GPGPUState *s)
{
    VramAllocator *a = &s->vram_alloc;

    /* destroy any existing pages (for reset) */
    while (a->pages)
        destroyPage(a, a->pages);

    memset(a, 0, sizeof(*a));
    a->baseAddress = 0;
    a->capacity = (uint32_t)s->vram_size;
    a->pageAlign = VRAM_PAGE_ALIGN;
    a->blockAlign = VRAM_BLOCK_ALIGN;

    /* reserve param area */
    vram_reserve(s, 0, 0x1000);
    /* reserve kernel code area */
    vram_reserve(s, 0x500000, 0x10000);

    memset(s->vram_ptr + PTR_TABLE_OFFSET, 0, PTR_TABLE_SIZE);
}

void vram_alloc_reset(struct GPGPUState *s)
{
    vram_alloc_init(s);
}

uint32_t vram_alloc(struct GPGPUState *s, size_t size)
{
    VramAllocator *a = &s->vram_alloc;
    if (size == 0) return 0;

    uint32_t asize = alignSize((uint32_t)size, a->blockAlign);

    /* search existing pages for a free block (best-fit) */
    VramBlock *fb = NULL;
    VramPage *page = a->pages;
    while (page) {
        fb = findFreeBlock(page, asize);
        if (fb) break;
        page = page->next;
    }

    /* no existing block found → create a new page */
    if (!fb) {
        uint32_t pageSize = alignSize(asize, a->pageAlign);
        uint32_t pageAddr;
        if (!findNextAddress(a, pageSize, &pageAddr)) {
            fprintf(stderr, "vram_alloc: OOM (need %u bytes)\n", (unsigned)size);
            return 0;
        }
        page = createPage(a, pageAddr, pageSize);
        if (!page) return 0;
        fb = findFreeBlock(page, asize);
        if (!fb) return 0;
    }

    pageAllocate(page, asize, fb);
    a->allocated += asize;
    return fb->addr;
}

void vram_free(struct GPGPUState *s, uint32_t addr)
{
    VramAllocator *a = &s->vram_alloc;
    VramPage *page = a->pages;
    while (page) {
        VramBlock *b = findUsedBlock(page, addr);
        if (b) {
            uint32_t sz = b->size;
            pageRelease(page, b);
            a->allocated -= sz;
            if (!page->usedList) destroyPage(a, page);
            return;
        }
        page = page->next;
    }
}

void vram_reserve(struct GPGPUState *s, uint32_t addr, size_t size)
{
    VramAllocator *a = &s->vram_alloc;
    if (size == 0) return;

    uint32_t asize = alignSize((uint32_t)size, a->pageAlign);

    if (hasPageOverlap(a, addr, asize)) {
        fprintf(stderr, "vram_reserve: [0x%x-0x%x] overlaps existing allocation\n", addr, addr + (uint32_t)size);
        return;
    }

    VramPage *page = createPage(a, addr, asize);
    if (!page) return;

    VramBlock *fb = findFreeBlock(page, asize);
    if (fb) {
        pageAllocate(page, asize, fb);
        a->allocated += asize;
    }
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
