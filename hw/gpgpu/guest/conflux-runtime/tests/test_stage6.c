#define _DEFAULT_SOURCE

#include "conflux_cmd_builder.h"
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

/* 模拟设备执行回调 */
int fake_execute(conflux_cmd_t *cmd, void *user_data) 
{
    (void)user_data;
    
    const char *type_str = "?";
    switch (cmd->type) {
        case CONFLUX_CMD_COPY:    type_str = "COPY";    break;
        case CONFLUX_CMD_KERNEL:  type_str = "KERNEL";  break;
        case CONFLUX_CMD_ALLOC:   type_str = "ALLOC";   break;
        case CONFLUX_CMD_FREE:    type_str = "FREE";    break;
        case CONFLUX_CMD_BARRIER: type_str = "BARRIER"; break;
        case CONFLUX_CMD_NOP:     type_str = "NOP";     break;
    }
    
    printf("  [DEVICE] Executing %s\n", type_str);
    usleep(10000);  /* 模拟 10ms */
    
    return 0;
}

int main(void) 
{
    printf("=== Stage 6: Command Builder ===\n\n");
    
    /* 初始化所有组件 */
    printf("--- Setup ---\n");
    
    /* 分配器：64KB 显存，4KB 块，基地址 0x10000000 */
    conflux_allocator_t alloc;
    int ret = conflux_allocator_init(&alloc, 0x10000000, 64 * 1024, 4 * 1024);
    assert(ret == 0);
    
    /* 队列：8 个条目，用 fake_execute 作为消费者 */
    conflux_queue_t *queue = conflux_queue_create(8, fake_execute, NULL);
    assert(queue != NULL);
    
    /* 启动消费者线程 */
    conflux_queue_start_consumer(queue);
    
    /* 构建器 */
    conflux_cmd_builder_t *builder = conflux_cmd_builder_create(queue, &alloc);
    assert(builder != NULL);
    
    printf("  All components ready\n\n");
    
    /* 测试 1：分配两块显存 */
    printf("--- Test 1: Alloc two buffers ---\n");
    uint64_t buf_a, buf_b;
    
    ret = conflux_cmd_builder_alloc(builder, 4096, &buf_a, NULL);
    assert(ret == CONFLUX_SUCCESS);
    printf("  buf_a = 0x%lx\n", (unsigned long)buf_a);
    
    ret = conflux_cmd_builder_alloc(builder, 8192, &buf_b, NULL);
    assert(ret == CONFLUX_SUCCESS);
    printf("  buf_b = 0x%lx\n", (unsigned long)buf_b);
    
    conflux_allocator_dump(&alloc);
    printf("  OK\n\n");
    
    /* 测试 2：内存拷贝 */
    printf("--- Test 2: Copy buffer ---\n");
    conflux_copy_request_t copy_req = {
        .src_addr = buf_a,
        .dst_addr = buf_b,
        .size     = 4096,
    };
    
    conflux_event_t *copy_event = NULL;
    ret = conflux_cmd_builder_copy(builder, &copy_req, &copy_event);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_event_wait(copy_event, 0);
    printf("  Copy completed\n");
    conflux_event_destroy(copy_event);
    printf("  OK\n\n");
    
    /* 测试 3：上传并执行内核 */
    printf("--- Test 3: Upload and launch kernel ---\n");
    
    /* 做一个假指令 */
    uint8_t dummy_instr[] = {0x00, 0x01, 0x02, 0x03, 0x04};
    
    conflux_kernel_t *kernel = conflux_kernel_create("vec_add", 1,
                                                    dummy_instr,
                                                    sizeof(dummy_instr));
    assert(kernel != NULL);
    
    /* 设置参数 */
    conflux_kernel_set_arg(kernel, 0, sizeof(uint64_t), &buf_a, 0);
    conflux_kernel_set_arg(kernel, 1, sizeof(uint64_t), &buf_b, 0);
    conflux_kernel_set_arg(kernel, 2, sizeof(uint32_t), &(uint32_t){1024}, 0);
    
    /* 上传指令 */
    uint64_t instr_addr;
    ret = conflux_cmd_builder_upload_kernel(builder, kernel, &instr_addr);
    assert(ret == CONFLUX_SUCCESS);
    printf("  Instructions at 0x%lx\n", (unsigned long)instr_addr);
    
    /* 启动内核 */
    conflux_launch_request_t launch_req = {
        .kernel    = kernel,
        .work_dim  = 1,
        .global_size = {1024, 1, 1},
        .local_size  = {64, 1, 1},
    };
    
    conflux_event_t *launch_event = NULL;
    ret = conflux_cmd_builder_launch(builder, &launch_req, &launch_event);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_event_wait(launch_event, 0);
    printf("  Kernel completed\n");
    conflux_event_destroy(launch_event);
    printf("  OK\n\n");
    
    /* 测试 4：屏障 */
    printf("--- Test 4: Barrier ---\n");
    conflux_event_t *barrier_event = NULL;
    ret = conflux_cmd_builder_barrier(builder, &barrier_event);
    assert(ret == CONFLUX_SUCCESS);
    conflux_event_wait(barrier_event, 0);
    printf("  Barrier passed\n");
    conflux_event_destroy(barrier_event);
    printf("  OK\n\n");
    
    /* 测试 5：释放显存 */
    printf("--- Test 5: Free buffers ---\n");
    ret = conflux_cmd_builder_free(builder, buf_a, 4096, NULL);
    assert(ret == CONFLUX_SUCCESS);
    ret = conflux_cmd_builder_free(builder, buf_b, 8192, NULL);
    assert(ret == CONFLUX_SUCCESS);
    
    conflux_allocator_dump(&alloc);
    printf("  OK\n\n");
    
    /* 统计 */
    conflux_cmd_builder_dump(builder);
    conflux_queue_dump(queue);
    
    /* 清理 */
    conflux_kernel_destroy(kernel);
    conflux_cmd_builder_destroy(builder);
    conflux_queue_stop_consumer(queue);
    conflux_queue_destroy(queue);
    conflux_allocator_destroy(&alloc);
    
    printf("\n=== Stage 6: ALL PASSED ===\n");
    return 0;
}