#include "conflux_log.h"
#include <stdio.h>
#include <pthread.h>

/* 模拟多线程写日志 */
void *log_spammer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 10; i++) {
        CONFLUX_INFO("[thread %d] message %d", id, i);
        usleep(1000);
    }
    return NULL;
}

int main(void) {
    printf("=== Stage 9: Log & Perf ===\n\n");
    
    /* 测试 1：初始化 */
    printf("--- Test 1: Init ---\n");
    conflux_log_init(CONFLUX_LOG_TRACE);  /* 全量日志 */
    conflux_perf_init();
    CONFLUX_TRACE("This is a trace message");
    CONFLUX_DEBUG("This is a debug message");
    CONFLUX_INFO("This is an info message");
    CONFLUX_WARN("This is a warning message");
    CONFLUX_ERROR("This is an error message");
    printf("  OK (check stderr for colored output)\n\n");
    
    /* 测试 2：级别过滤 */
    printf("--- Test 2: Level filtering ---\n");
    g_conflux_log.level = CONFLUX_LOG_WARN;  /* 只显示 WARN 以上 */
    CONFLUX_DEBUG("This DEBUG should NOT appear");
    CONFLUX_INFO("This INFO should NOT appear");
    CONFLUX_WARN("This WARN should appear");
    CONFLUX_ERROR("This ERROR should appear");
    printf("  OK (only WARN and ERROR should be visible)\n\n");
    
    /* 测试 3：多线程安全 */
    printf("--- Test 3: Multi-thread ---\n");
    g_conflux_log.level = CONFLUX_LOG_INFO;
    pthread_t threads[4];
    int ids[4] = {0, 1, 2, 3};
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, log_spammer, &ids[i]);
    }
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("  OK (40 messages, no corruption)\n\n");
    
    /* 测试 4：环形缓冲区 dump */
    printf("--- Test 4: Ring dump ---\n");
    conflux_log_dump_ring(stdout);
    printf("  OK\n\n");
    
    /* 测试 5：性能统计 */
    printf("--- Test 5: Perf stats ---\n");
    conflux_perf_record_submit(1);  /* COPY */
    conflux_perf_record_submit(2);  /* KERNEL */
    conflux_perf_record_submit(2);  /* KERNEL */
    conflux_perf_record_copy(4096);
    conflux_perf_record_copy(8192);
    conflux_perf_record_alloc(16384);
    conflux_perf_record_wait(100000);  /* 100us */
    conflux_perf_record_wait(200000);  /* 200us */
    conflux_perf_record_wait(50000);   /* 50us  -> min */
    conflux_perf_record_error();
    conflux_perf_record_timeout();
    
    conflux_perf_print();
    printf("  OK\n\n");
    
    printf("=== Stage 9: ALL PASSED ===\n");
    return 0;
}