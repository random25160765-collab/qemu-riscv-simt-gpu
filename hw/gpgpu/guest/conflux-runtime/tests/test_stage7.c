#include "conflux_device.h"
#include "conflux_cmd_builder.h"
#include <stdio.h>
#include <assert.h>

/* 模拟设备执行回调 */
int fake_execute(conflux_cmd_t *cmd, void *user_data) 
{
    (void)user_data;
    printf("  [DEVICE] Executing cmd type=%d\n", cmd->type);
    return 0;
}

int main(void) 
{
    conflux_error_t err;
    printf("=== Stage 7: Device Context ===\n\n");
    
    /* 测试 1：创建设备对象 */
    printf("--- Test 1: Create device ---\n");
    conflux_device_t *dev = conflux_device_create();
    assert(dev != NULL);
    conflux_device_dump(dev);
    printf("  OK\n\n");
    
    /* 测试 2：初始化（模拟模式） */
    printf("--- Test 2: Initialize (simulation) ---\n");
    int ret = conflux_device_init(dev,
                                  NULL,           /* 无真实设备 */
                                  0x10000000,     /* MMIO 基地址 */
                                  64 * 1024 * 1024);  /* 64 MB */
    assert(ret == CONFLUX_SUCCESS);
    conflux_device_dump(dev);
    printf("  OK\n\n");
    
    /* 测试 3：上线 */
    printf("--- Test 3: Bring online ---\n");
    ret = conflux_device_online(dev);
    assert(ret == CONFLUX_SUCCESS);
    assert(conflux_device_is_online(dev) == 1);
    printf("  Online: %d\n", conflux_device_is_online(dev));
    printf("  OK\n\n");
    
    /* 测试 4：用分配器分配内存 */
    printf("--- Test 4: Use allocator via device ---\n");
    conflux_allocator_t *alloc = conflux_device_get_allocator(dev);
    assert(alloc != NULL);
    
    uint64_t addr1 = conflux_allocator_alloc(alloc, 8 * 1024);  /* 8KB */
    assert(addr1 != UINT64_MAX);
    printf("  Allocated 8KB at 0x%lx\n", (unsigned long)addr1);

    uint64_t addr2 = conflux_allocator_alloc(alloc, 16 * 1024); /* 16KB */
    (void)err;
    assert(addr2 != UINT64_MAX);
    printf("  Allocated 16KB at 0x%lx\n", (unsigned long)addr2);
    
    conflux_device_dump(dev);
    printf("  OK\n\n");
    
    /* 测试 5：reset 后状态恢复 */
    printf("--- Test 5: Reset ---\n");
    ret = conflux_device_reset(dev);
    assert(ret == CONFLUX_SUCCESS);
    conflux_device_dump(dev);
    printf("  OK\n\n");
    
    /* 测试 6：查询信息 */
    printf("--- Test 6: Query info ---\n");
    char info[1024];
    conflux_device_query_info(dev, info, sizeof(info));
    printf("%s", info);
    printf("  OK\n\n");
    
    /* 测试 7：离线/销毁 */
    printf("--- Test 7: Offline and destroy ---\n");
    ret = conflux_device_offline(dev);
    assert(ret == CONFLUX_SUCCESS);
    assert(conflux_device_is_online(dev) == 0);
    
    conflux_device_destroy(dev);
    printf("  OK\n\n");
    
    printf("=== Stage 7: ALL PASSED ===\n");
    return 0;
}