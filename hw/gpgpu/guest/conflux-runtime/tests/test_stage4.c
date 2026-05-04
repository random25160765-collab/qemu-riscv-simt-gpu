#include "conflux_queue.h"
#include "conflux_event.h"
#include <stdio.h>
#include <unistd.h>

/* 模拟设备：简单地 sleep 一下表示执行 */
int fake_execute(conflux_cmd_t *cmd, void *user_data) 
{
    (void)user_data;  /* 消除未使用参数警告 */
    const char *type_str = "UNKNOWN";
    switch (cmd->type) {
        case CONFLUX_CMD_NOP:     type_str = "NOP";     break;
        case CONFLUX_CMD_COPY:    type_str = "COPY";    break;
        case CONFLUX_CMD_KERNEL:  type_str = "KERNEL";  break;
        case CONFLUX_CMD_ALLOC:   type_str = "ALLOC";   break;
        case CONFLUX_CMD_FREE:    type_str = "FREE";    break;
        case CONFLUX_CMD_BARRIER: type_str = "BARRIER"; break;
    }
    
    printf("[DEVICE] Executing %s cmd (src=0x%lx, dst=0x%lx, size=%u)\n",
           type_str, (unsigned long)cmd->src_addr, 
           (unsigned long)cmd->dst_addr, cmd->size);
    
    usleep(10000);  /* 模拟 10ms 执行时间 */
    
    return 0;
}

int main(void) 
{
    printf("=== Stage 4: Command Queue ===\n\n");
    
    /* 测试 1：创建/销毁 */
    printf("--- Test 1: Create and destroy ---\n");
    conflux_queue_t *q = conflux_queue_create(8, fake_execute, NULL);
    if (!q) {
        printf("  FAIL: create failed\n");
        return 1;
    }
    conflux_queue_dump(q);
    conflux_queue_destroy(q);
    printf("  OK\n\n");
    
    /* 测试 2：提交单个命令（同步） */
    printf("--- Test 2: Submit single command (sync) ---\n");
    q = conflux_queue_create(8, fake_execute, NULL);
    conflux_queue_start_consumer(q);
    
    conflux_cmd_t cmd = {
        .type     = CONFLUX_CMD_COPY,
        .src_addr = 0x1000,
        .dst_addr = 0x2000,
        .size     = 1024,
    };
    
    int ret = conflux_queue_submit_sync(q, &cmd);
    printf("  submit_sync returned: %d (expect 0)\n", ret);
    printf("  completed: %u (expect 1)\n", conflux_queue_get_completed(q));  // ← 修改
    
    conflux_queue_dump(q);
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("  OK\n\n");
    
    /* 测试 3：提交多个命令（异步），用事件等最后一个 */
    printf("--- Test 3: Submit multiple commands (async) ---\n");
    q = conflux_queue_create(8, fake_execute, NULL);
    conflux_queue_start_consumer(q);
    
    conflux_event_t *last_event = NULL;
    
    for (int i = 0; i < 5; i++) {
        conflux_cmd_t c = {
            .type     = CONFLUX_CMD_KERNEL,
            .src_addr = 0x10000 + i * 0x1000,
            .size     = 256,
            .kernel_id = i,
        };
        
        ret = conflux_queue_submit(q, &c, (i == 4) ? &last_event : NULL);
        if (ret != CONFLUX_SUCCESS) {
            printf("  FAIL: submit %d returned %d\n", i, ret);
            return 1;
        }
    }
    
    printf("  Submitted 5 commands, waiting for last...\n");
    conflux_event_wait(last_event, 0);
    printf("  Last event completed\n");
    
    printf("  completed: %u (expect 5)\n", conflux_queue_get_completed(q));  // ← 修改
    
    conflux_event_destroy(last_event);
    conflux_queue_dump(q);
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("  OK\n\n");
    
    /* 测试 4：队列满阻塞 */
    printf("--- Test 4: Queue full backpressure ---\n");
    q = conflux_queue_create(4, fake_execute, NULL);  /* 小队列 */
    conflux_queue_start_consumer(q);
    
    printf("  Submitting 8 commands (queue size is 4)...\n");
    for (int i = 0; i < 8; i++) {
        conflux_cmd_t c = {
            .type = CONFLUX_CMD_NOP,
        };
        ret = conflux_queue_submit_sync(q, &c);
        if (ret != CONFLUX_SUCCESS) {
            printf("  FAIL: submit %d returned %d\n", i, ret);
            return 1;
        }
    }
    
    printf("  All 8 completed\n");
    printf("  completed: %u (expect 8)\n", conflux_queue_get_completed(q));  // ← 修改
    
    conflux_queue_dump(q);
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("  OK\n\n");
    
    /* 测试 5：排空 */
    printf("--- Test 5: Drain ---\n");
    q = conflux_queue_create(8, fake_execute, NULL);
    conflux_queue_start_consumer(q);
    
    for (int i = 0; i < 3; i++) {
        conflux_cmd_t c = { .type = CONFLUX_CMD_COPY, .size = 64 };
        conflux_queue_submit(q, &c, NULL);
    }
    
    printf("  Submitted 3 commands, draining...\n");
    ret = conflux_queue_drain(q);
    printf("  Drain returned: %d (expect 0)\n", ret);
    printf("  pending: %u (expect 0)\n", conflux_queue_pending_count(q));
    
    conflux_queue_dump(q);
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("  OK\n\n");
    
    printf("=== Stage 4: ALL PASSED ===\n");
    return 0;
}