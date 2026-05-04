#include "conflux_allocator.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

static int bitmap_get(const bitmap_word_t *bitmap, uint32_t n)
{
    return (bitmap[BITMAP_WORD(n)] >> BITMAP_BIT(n)) & 1;
}

static void bitmap_set(bitmap_word_t *bitmap, uint32_t n)
{
    bitmap[BITMAP_WORD(n)] |= (1U << BITMAP_BIT(n));
}

static void bitmap_clear(bitmap_word_t *bitmap, uint32_t n)
{
    bitmap[BITMAP_WORD(n)] &= ~(1U << BITMAP_BIT(n)); 
}

/* find {num} contiguous spare blocks and return the start */
static int bitmap_find_contiguous(const bitmap_word_t *bitmap,
                                  uint32_t total_blocks, 
                                  uint32_t num)
{
    uint32_t start = 0;
    uint32_t count = 0;
    
    for (uint32_t i = 0; i < total_blocks; i++) {
        if (bitmap_get(bitmap, i) == 0) {
            /* find the start */
            if (count == 0) {
                start = i;
            }
            count++;
            /* success */
            if (count == num) {
                return start;
            }
        } else {
            /* reset the count */
            count = 0;
        }
    }
    /* failed to find */
    return -1;
}


int conflux_allocator_init(conflux_allocator_t *alloc,
                          uint64_t base_addr,
                          uint64_t total_size,
                          uint32_t block_size)
{
    /* check the param */
    if ((block_size & (block_size - 1)) != 0) {
        CONFLUX_ERROR("[ALLOC] ERROR: block_size must be power of 2\n");
        return -1;
    }
    if (total_size < block_size) {
        CONFLUX_ERROR("[ALLOC] ERROR: total_size < block_size\n");
        return -2;
    }

    alloc->base_addr    = base_addr;
    alloc->total_size   = total_size;
    alloc->block_size   = block_size;
    alloc->total_blocks = total_size / block_size;

    size_t bitmap_bytes = ((alloc->total_blocks + BITS_PER_WORD - 1) / BITS_PER_WORD) * sizeof(bitmap_word_t);
    alloc->bitmap = malloc(bitmap_bytes);
    if (!alloc->bitmap) {
        return -3;
    }

    memset(alloc->bitmap, 0, bitmap_bytes);
    alloc->free_blocks = alloc->total_blocks;

    CONFLUX_INFO("[ALLOC] Init: base=0x%lx, total=%lu, block=%u, blocks=%u\n",
                (unsigned long)base_addr, (unsigned long)total_size,
                block_size, alloc->total_blocks);

    return 0;
}

void conflux_allocator_destroy(conflux_allocator_t *alloc)
{
    if (alloc->bitmap) {
        free(alloc->bitmap);
        alloc->bitmap = NULL;
    }
}

uint64_t conflux_allocator_alloc(conflux_allocator_t *alloc, size_t size)
{
    uint32_t blocks_needed = (size + alloc->block_size - 1)
                             / alloc->block_size;
    
    /* 怎么可能是0呢？ */
    if (blocks_needed == 0) {
        blocks_needed = 1;
    }

    if (blocks_needed > alloc->free_blocks) {
        CONFLUX_ERROR("[ALLOC] Not enough free blocks: need %u, have %u\n",
                     blocks_needed, alloc->free_blocks);
        return UINT64_MAX;
    }

    /* 查找连续空闲块 */
    int start = bitmap_find_contiguous(alloc->bitmap, 
                                       alloc->total_blocks, 
                                       blocks_needed);
    if (start < 0) {
        CONFLUX_ERROR("[ALLOC] Fragmentation: no %u contiguous blocks\n",
                     blocks_needed);
        return UINT64_MAX;
    }

    /* 标记这些块为已占用 */
    for (uint32_t i = start; i < start + blocks_needed; i++) {
        bitmap_set(alloc->bitmap, i);
    }
    alloc->free_blocks -= blocks_needed;

    uint64_t dev_addr = alloc->base_addr + ((uint64_t)start * alloc->block_size);

    CONFLUX_DEBUG("[ALLOC] Allocated %zu bytes at 0x%lx (blocks %u-%u)\n",
                 size, (unsigned long)dev_addr, start, start + blocks_needed - 1);
    
    return dev_addr;
}

int conflux_allocator_free(conflux_allocator_t *alloc,
                          uint64_t dev_addr,
                          size_t size) 
{
    if (dev_addr < alloc->base_addr) {
        return -1;  /* 地址不在管理范围内 */
    }
    
    /* 计算起始块编号 */
    uint32_t start_block = (dev_addr - alloc->base_addr) 
                           / alloc->block_size;
    uint32_t blocks_to_free = (size + alloc->block_size - 1) 
                              / alloc->block_size;
    
    if (start_block + blocks_to_free > alloc->total_blocks) {
        return -2;  /* 超出范围 */
    }
    
    /* 清零位图 */
    for (uint32_t i = start_block; i < start_block + blocks_to_free; i++) {
        bitmap_clear(alloc->bitmap, i);
    }
    alloc->free_blocks += blocks_to_free;
    
    CONFLUX_DEBUG("[ALLOC] Freed %zu bytes at 0x%lx (blocks %u-%u)\n",
                 size, (unsigned long)dev_addr, 
                 start_block, start_block + blocks_to_free - 1);
    
    return 0;
}

/* dump 函数的 printf 保留，不做替换 */
void conflux_allocator_dump(const conflux_allocator_t *alloc) 
{
    printf("\n=== Allocator State ===\n");
    printf("  base:   0x%016lx\n", (unsigned long)alloc->base_addr);
    printf("  total:  %lu bytes\n", (unsigned long)alloc->total_size);
    printf("  block:  %u bytes\n", alloc->block_size);
    printf("  blocks: %u total, %u free, %u used\n",
           alloc->total_blocks,
           alloc->free_blocks,
           alloc->total_blocks - alloc->free_blocks);
    
    /* 打印位图（前 64 块） */
    printf("  bitmap (first 64): ");
    for (uint32_t i = 0; i < 64 && i < alloc->total_blocks; i++) {
        printf("%d", bitmap_get(alloc->bitmap, i));
    }
    printf("\n");
}