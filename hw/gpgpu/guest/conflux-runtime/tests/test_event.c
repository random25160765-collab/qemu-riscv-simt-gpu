#include "conflux_event.h"
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>   /* usleep */

/* 异步模拟：另一个线程在 200ms 后完成事件 */
void *device_simulator(void *arg) 
{
    conflux_event_t *event = (conflux_event_t *)arg;
    
    printf("[SIM] Device received event, processing...\n");
    usleep(50000);  /* 模拟设备 50ms 处理时间 */
    
    printf("[SIM] Device setting event to RUNNING\n");
    conflux_event_set_running(event);
    
    usleep(150000); /* 再花 150ms */
    
    printf("[SIM] Device setting event to COMPLETE\n");
    conflux_event_set_complete(event);
    
    return NULL;
}

/* 完成回调 */
void on_complete(conflux_event_t *event, void *data) 
{
    printf("[CALLBACK] Event %p completed! user_data=%s\n",
           (void *)event, (const char *)data);
}

int main(void) 
{
    printf("=== Stage 3: Event Object ===\n\n");
    
    /* 测试 1：创建与销毁 */
    printf("--- Test 1: Create and destroy ---\n");
    conflux_event_t *ev1 = conflux_event_create();
    conflux_event_destroy(ev1);
    printf("  OK\n\n");
    
    /* 测试 2：状态转换 */
    printf("--- Test 2: Status transitions ---\n");
    conflux_event_t *ev2 = conflux_event_create();
    
    printf("Initial status: %d (expect QUEUED=0)\n", 
           conflux_event_get_status(ev2));
    
    conflux_event_set_submitted(ev2);
    printf("After submit: %d (expect SUBMITTED=1)\n",
           conflux_event_get_status(ev2));
    
    conflux_event_set_running(ev2);
    printf("After running: %d (expect RUNNING=2)\n",
           conflux_event_get_status(ev2));
    
    conflux_event_set_complete(ev2);
    printf("After complete: %d (expect COMPLETE=3)\n",
           conflux_event_get_status(ev2));
    
    printf("Is complete? %d (expect 1)\n", conflux_event_is_complete(ev2));
    
    conflux_event_destroy(ev2);
    printf("  OK\n\n");
    
    /* 测试 3：异步等待 */
    printf("--- Test 3: Async wait ---\n");
    conflux_event_t *ev3 = conflux_event_create();
    
    conflux_event_set_submitted(ev3);
    
    pthread_t sim_thread;
    pthread_create(&sim_thread, NULL, device_simulator, ev3);
    
    printf("[MAIN] Waiting for event to complete...\n");
    int ret = conflux_event_wait(ev3, 0);  /* 0 = 无限等待 */
    printf("[MAIN] Wait returned: %d (expect 0)\n", ret);
    printf("[MAIN] Event status: %d (expect COMPLETE=3)\n",
           conflux_event_get_status(ev3));
    
    pthread_join(sim_thread, NULL);
    conflux_event_destroy(ev3);
    printf("  OK\n\n");
    
    /* 测试 4：超时等待 */
    printf("--- Test 4: Timeout ---\n");
    conflux_event_t *ev4 = conflux_event_create();
    conflux_event_set_submitted(ev4);
    /* 不完成它，只等 100ms */
    
    printf("[MAIN] Waiting 100ms for event that never completes...\n");
    ret = conflux_event_wait(ev4, 100000000ULL);  /* 100ms = 100,000,000 ns */
    printf("[MAIN] Wait returned: %d (expect 1=TIMEOUT)\n", ret);
    
    conflux_event_set_complete(ev4);  /* 手动完成 */
    conflux_event_destroy(ev4);
    printf("  OK\n\n");
    
    /* 测试 5：回调 */
    printf("--- Test 5: Callback ---\n");
    conflux_event_t *ev5 = conflux_event_create();
    conflux_event_set_callback(ev5, on_complete, "my_data");
    conflux_event_set_submitted(ev5);
    conflux_event_set_complete(ev5);  /* 应该触发 on_complete */
    conflux_event_destroy(ev5);
    printf("  OK\n\n");
    
    /* 测试 6：引用计数 */
    printf("--- Test 6: Reference counting ---\n");
    conflux_event_t *ev6 = conflux_event_create();
    conflux_event_retain(ev6);
    conflux_event_release(ev6);  /* ref=1，不销毁 */
    conflux_event_release(ev6);  /* ref=0，销毁 */
    printf("  OK (no crash)\n\n");
    
    /* 测试 7：dump */
    printf("--- Test 7: Debug dump ---\n");
    conflux_event_t *ev7 = conflux_event_create();
    char buf[512];
    conflux_event_dump(ev7, buf, sizeof(buf));
    printf("%s\n", buf);
    conflux_event_destroy(ev7);
    printf("  OK\n\n");
    
    printf("=== Stage 3: ALL PASSED ===\n");
    return 0;
}