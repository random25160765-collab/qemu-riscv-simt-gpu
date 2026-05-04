#include "conflux_allocator.h"
#include <stdio.h>
#include <assert.h>

int main(void) 
{
    conflux_allocator_t alloc;
    uint64_t addr1, addr2, addr3, addr4;

    printf("=== Bitmap Allocator Test ===\n\n");
    
    /* 初始化：基地址 0x1000000，64KB 显存，每块 4KB */
    int ret = conflux_allocator_init(&alloc, 
                                    0x1000000,  /* 16MB 起始 */
                                    64 * 1024,   /* 64KB */
                                    4 * 1024);   /* 4KB 块 */
    assert(ret == 0);
    conflux_allocator_dump(&alloc);
    
    /* 测试 1：分配小内存（刚好 1 块） */
    printf("\nTest 1: Alloc 4KB (1 block)\n");
    addr1 = conflux_allocator_alloc(&alloc, 4 * 1024);
    assert(addr1 != UINT64_MAX);
    assert(addr1 == 0x1000000);
    conflux_allocator_dump(&alloc);

    /* 测试 2：分配 1 字节（还是占 1 块） */
    printf("\nTest 2: Alloc 1 byte (still 1 block)\n");
    addr2 = conflux_allocator_alloc(&alloc, 1);
    assert(addr2 != UINT64_MAX);
    assert(addr2 == 0x1001000);  /* 下一块 */
    conflux_allocator_dump(&alloc);
    
    /* 测试 3：分配 10KB（3 块） */
    printf("\nTest 3: Alloc 10KB (3 blocks)\n");
    addr3 = conflux_allocator_alloc(&alloc, 10 * 1024);
    assert(addr3 != UINT64_MAX);
    assert(addr3 == 0x1002000);
    conflux_allocator_dump(&alloc);
    
    /* 测试 4：释放 addr2（1字节那块），制造碎片 */
    printf("\nTest 4: Free addr2 (creates hole)\n");
    ret = conflux_allocator_free(&alloc, addr2, 1);
    assert(ret == 0);
    conflux_allocator_dump(&alloc);
    
    /* 测试 5：再次分配 4KB，应该能填回那个洞 */
    printf("\nTest 5: Alloc 4KB, should reuse hole\n");
    addr4 = conflux_allocator_alloc(&alloc, 4 * 1024);
    assert(addr4 != UINT64_MAX);
    assert(addr4 == 0x1001000);  /* 正好是释放的 addr2 */
    conflux_allocator_dump(&alloc);
    
    /* 测试 6：分配过大导致失败 */
    printf("\nTest 6: Alloc 100KB (too big, expect fail)\n");
    uint64_t addr_fail = conflux_allocator_alloc(&alloc, 100 * 1024);
    assert(addr_fail == UINT64_MAX);
    printf("  Correctly returned UINT64_MAX\n");
    
    /* 清理 */
    conflux_allocator_destroy(&alloc);
    
    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}