#ifndef CONFLUX_ALLOCATOR_H
#define CONFLUX_ALLOCATOR_H

#include "conflux_error.h"
#include <stdint.h>
#include <stddef.h>

/* 第一个设计点：用 uint32 数组存储内存状态，每一个数存储32个块的状态 */
typedef uint32_t bitmap_word_t;

#define BITS_PER_WORD   32
/* n 是绝对位置，用 “第几个 word 的第几个 bit 来描述” */
#define BITMAP_WORD(n)  ((n) / BITS_PER_WORD)
#define BITMAP_BIT(n)   ((n) & (BITS_PER_WORD - 1))

typedef struct {
    uint64_t base_addr;
    uint64_t total_size;   
    /* Pay attention: the block size must be the power of 2 */
    uint32_t block_size;
    uint32_t total_blocks;
    bitmap_word_t *bitmap;
    uint32_t free_blocks;
} conflux_allocator_t;

int conflux_allocator_init(conflux_allocator_t *alloc,
                          uint64_t base_addr,
                          uint64_t total_size,
                          uint32_t block_size);

void conflux_allocator_destroy(conflux_allocator_t *alloc);

uint64_t conflux_allocator_alloc(conflux_allocator_t *alloc, size_t size);

int conflux_allocator_free(conflux_allocator_t *alloc,
                          uint64_t dev_addr,
                          size_t size);

void conflux_allocator_dump(const conflux_allocator_t *alloc);

#endif