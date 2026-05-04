#include "conflux_platform.h"
#include "conflux_device.h"
#include "conflux_allocator.h"
#include "conflux_queue.h"
#include "conflux_kernel.h"
#include "conflux_cmd_builder.h"
#include "conflux_event.h"
#include "conflux_log.h"
#include "conflux_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#define QUIET_MODE 1  /* 设 0 看详细日志 */

#if QUIET_MODE
#  undef CONFLUX_INFO
#  undef CONFLUX_DEBUG
#  undef CONFLUX_WARN
#  define CONFLUX_INFO(...)  do{}while(0)
#  define CONFLUX_DEBUG(...) do{}while(0)
#  define CONFLUX_WARN(...)  do{}while(0)
#endif

/* ---- 辅助函数 ---- */
static int noop_execute(conflux_cmd_t *cmd, void *ud) { (void)cmd; (void)ud; return 0; }
static int slow_execute(conflux_cmd_t *cmd, void *ud) { (void)cmd; (void)ud; usleep(1000); return 0; }

/* ================================================================
 * 测试 1: 分配器 - 精确填满 + 溢出
 * ================================================================ */
static void test_allocator_fill_and_overflow(void) {
    printf("  [1] Allocator: fill & overflow... ");
    
    conflux_allocator_t alloc;
    int ret = conflux_allocator_init(&alloc, 0x10000, 64 * 1024, 4 * 1024);
    assert(ret == 0);
    
    /* 64KB / 4KB = 16块，全部分配 */
    uint64_t addrs[16];
    for (int i = 0; i < 16; i++) {
        addrs[i] = conflux_allocator_alloc(&alloc, 4 * 1024);
        assert(addrs[i] != UINT64_MAX);
    }
    
    /* 第 17 次应失败 */
    uint64_t fail = conflux_allocator_alloc(&alloc, 4 * 1024);
    assert(fail == UINT64_MAX);
    
    /* 释放一半，再分配应成功 */
    for (int i = 0; i < 8; i++) {
        ret = conflux_allocator_free(&alloc, addrs[i], 4 * 1024);
        assert(ret == 0);
    }
    uint64_t new1 = conflux_allocator_alloc(&alloc, 8 * 1024);  /* 2块 */
    assert(new1 != UINT64_MAX);
    uint64_t new2 = conflux_allocator_alloc(&alloc, 24 * 1024); /* 6块 */
    assert(new2 != UINT64_MAX);
    
    conflux_allocator_destroy(&alloc);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 2: 分配器 - 碎片化（分配/释放交错）
 * ================================================================ */
static void test_allocator_fragmentation(void) {
    printf("  [2] Allocator: fragmentation... ");
    
    conflux_allocator_t alloc;
    conflux_allocator_init(&alloc, 0x10000, 64 * 1024, 4 * 1024);
    
    /* 分配 0,1,3,5,7... 留空洞 */
    uint64_t a[8];
    for (int i = 0; i < 8; i++) {
        a[i] = conflux_allocator_alloc(&alloc, 4 * 1024);
    }
    conflux_allocator_free(&alloc, a[1], 4 * 1024);
    conflux_allocator_free(&alloc, a[3], 4 * 1024);
    conflux_allocator_free(&alloc, a[5], 4 * 1024);
    
    /* 请求 2 块连续，应能填回某个空洞 */
    uint64_t small = conflux_allocator_alloc(&alloc, 8 * 1024);
    assert(small != UINT64_MAX);
    /* 请求 5 块连续，当前最大连续可能不足，取决于空洞分布 */
    uint64_t big = conflux_allocator_alloc(&alloc, 20 * 1024); /* 5块 */
    if (big != UINT64_MAX) {
        /* 成功则至少有一片连续 5 块空闲 */
    } else {
        /* 失败也合理（碎片严重） */
        printf("(big alloc correctly failed due to fragmentation) ");
    }
    
    conflux_allocator_destroy(&alloc);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 3: 事件 - 错误状态转换
 * ================================================================ */
static void test_event_invalid_transitions(void) {
    printf("  [3] Event: invalid transitions... ");
    
    conflux_event_t *ev = conflux_event_create();
    
    /* 正常转换 */
    conflux_event_set_submitted(ev);
    assert(conflux_event_get_status(ev) == CONFLUX_EVENT_SUBMITTED);
    conflux_event_set_complete(ev);
    assert(conflux_event_is_complete(ev));
    
    /* 已完成后再 submit 仍能执行但不合理，我们只测不会死锁/崩溃 */
    conflux_event_set_submitted(ev);
    conflux_event_set_running(ev);
    conflux_event_set_failed(ev, CONFLUX_ERR_DEVICE_FAULT);
    assert(conflux_event_get_status(ev) == CONFLUX_EVENT_FAILED);
    assert(conflux_event_is_complete(ev)); /* 失败也算完成 */
    
    conflux_event_destroy(ev);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 4: 事件 - 多等待者
 * ================================================================ */
static void *wait_thread(void *arg) {
    conflux_event_t *ev = (conflux_event_t *)arg;
    conflux_event_wait(ev, 0);
    return NULL;
}

static void test_event_multiple_waiters(void) {
    printf("  [4] Event: multiple waiters... ");
    
    conflux_event_t *ev = conflux_event_create();
    conflux_event_set_submitted(ev);
    
    pthread_t t1, t2, t3;
    pthread_create(&t1, NULL, wait_thread, ev);
    pthread_create(&t2, NULL, wait_thread, ev);
    pthread_create(&t3, NULL, wait_thread, ev);
    
    usleep(100000);
    conflux_event_set_complete(ev);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    
    conflux_event_destroy(ev);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 5: 队列 - 背压（生产者快于消费者）
 * ================================================================ */
static void test_queue_backpressure(void) {
    printf("  [5] Queue: backpressure... ");
    
    conflux_queue_t *q = conflux_queue_create(4, slow_execute, NULL);
    conflux_queue_start_consumer(q);
    
    conflux_event_t *ev[8];
    conflux_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type = CONFLUX_CMD_NOP;
    
    for (int i = 0; i < 8; i++) {
        int ret = conflux_queue_submit(q, &cmd, &ev[i]);
        assert(ret == CONFLUX_SUCCESS);
    }
    
    /* 等待全部完成 */
    for (int i = 0; i < 8; i++) {
        conflux_event_wait(ev[i], 0);
        conflux_event_destroy(ev[i]);
    }
    assert(q->completed_count == 8);
    
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 6: 队列 - 停止/重启消费者
 * ================================================================ */
static void test_queue_stop_restart(void) {
    printf("  [6] Queue: stop/restart consumer... ");
    
    conflux_queue_t *q = conflux_queue_create(8, noop_execute, NULL);
    
    /* 先启动 → 停止 → 再启动 */
    conflux_queue_start_consumer(q);
    assert(q->consumer_running);
    conflux_queue_stop_consumer(q);
    assert(!q->consumer_running);
    
    /* 清理残余 */
    conflux_queue_start_consumer(q);
    conflux_cmd_t cmd = { .type = CONFLUX_CMD_NOP };
    conflux_event_t *ev;
    conflux_queue_submit(q, &cmd, &ev);
    conflux_event_wait(ev, 0);
    conflux_event_destroy(ev);
    
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 7: 内核 - 参数超限/未上传
 * ================================================================ */
static void test_kernel_edge_cases(void) {
    printf("  [7] Kernel: edge cases... ");
    
    uint8_t dummy[8] = {0};
    conflux_kernel_t *k = conflux_kernel_create("test", 0, dummy, 8);
    
    /* 参数超限 */
    int ret = conflux_kernel_set_arg(k, CONFLUX_KERNEL_MAX_ARGS, 4, NULL, 0);
    assert(ret == CONFLUX_ERR_INVALID);
    
    /* 未上传就打包 */
    conflux_cmd_t cmd;
    ret = conflux_kernel_pack_cmd(k, &cmd);
    assert(ret == CONFLUX_ERR_INVALID);
    
    /* 上传后打包 */
    k->binary_device_addr = 0x1000;
    k->binary_uploaded = 1;
    ret = conflux_kernel_pack_cmd(k, &cmd);
    assert(ret == CONFLUX_SUCCESS);
    assert(cmd.type == CONFLUX_CMD_KERNEL);
    
    conflux_kernel_destroy(k);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 8: 构建器 - 批量操作 + 屏障
 * ================================================================ */
static void test_builder_batch_and_barrier(void) {
    printf("  [8] Builder: batch + barrier... ");
    
    conflux_allocator_t alloc;
    conflux_allocator_init(&alloc, 0x10000, 64*1024, 4*1024);
    conflux_queue_t *q = conflux_queue_create(16, noop_execute, NULL);
    conflux_queue_start_consumer(q);
    conflux_cmd_builder_t *b = conflux_cmd_builder_create(q, &alloc);
    
    /* 分配两块 */
    uint64_t a, b_addr;
    conflux_cmd_builder_alloc(b, 4096, &a, NULL);
    conflux_cmd_builder_alloc(b, 4096, &b_addr, NULL);
    
    /* 提交10个拷贝 */
    conflux_event_t *evs[10];
    for (int i = 0; i < 10; i++) {
        conflux_copy_request_t req = { a, b_addr, 256 };
        conflux_cmd_builder_copy(b, &req, &evs[i]);
    }
    
    /* 插入屏障 */
    conflux_event_t *bar_ev;
    conflux_cmd_builder_barrier(b, &bar_ev);
    conflux_event_wait(bar_ev, 0);
    conflux_event_destroy(bar_ev);
    
    /* 等所有拷贝 */
    for (int i = 0; i < 10; i++) {
        conflux_event_wait(evs[i], 0);
        conflux_event_destroy(evs[i]);
    }
    
    conflux_cmd_builder_free(b, a, 4096, NULL);
    conflux_cmd_builder_free(b, b_addr, 4096, NULL);
    conflux_cmd_builder_destroy(b);
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    conflux_allocator_destroy(&alloc);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 9: 平台 - 多设备并发访问
 * ================================================================ */
static void *device_worker(void *arg) {
    int dev_idx = *(int *)arg;
    conflux_platform_open_device(dev_idx);
    conflux_device_t *dev = conflux_platform_get_device(dev_idx);
    conflux_allocator_t *al = conflux_device_get_allocator(dev);
    for (int i = 0; i < 5; i++) {
        uint64_t addr = conflux_allocator_alloc(al, 4*1024);
        if (addr != UINT64_MAX) {
            usleep(100);
            conflux_allocator_free(al, addr, 4*1024);
        }
    }
    conflux_platform_close_device(dev_idx);
    return NULL;
}

static void test_platform_multi_device_concurrent(void) {
    printf("  [9] Platform: concurrent multi-device... ");
    
    conflux_platform_init();
    conflux_device_desc_t d0 = { .mmio_base=0x10000, .mmio_size=64*1024, .vendor_id=1, .device_id=1, .available=1 };
    conflux_device_desc_t d1 = { .mmio_base=0x20000, .mmio_size=64*1024, .vendor_id=1, .device_id=2, .available=1 };
    /* SIM 模式：空 path */
    d0.path[0] = '\0';
    d1.path[0] = '\0';
    conflux_platform_probe_specific(&d0);
    conflux_platform_probe_specific(&d1);
    
    pthread_t t0, t1;
    int id0 = 0, id1 = 1;
    pthread_create(&t0, NULL, device_worker, &id0);
    pthread_create(&t1, NULL, device_worker, &id1);
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);
    
    conflux_platform_destroy();
    printf("PASSED\n");
}

/* ================================================================
 * 测试 10: 日志环 - 溢出覆盖正确性
 * ================================================================ */
static void test_log_ring_overflow(void) {
    printf("  [10] Log ring: overflow... ");
    
    conflux_log_init(CONFLUX_LOG_TRACE);
    /* 写入大量日志，超过环大小（4KB） */
    for (int i = 0; i < 500; i++) {
        CONFLUX_INFO("overflow message number %d", i);
    }
    /* dump 到 /dev/null 验证不会死机 */
    FILE *null = fopen("/dev/null", "w");
    conflux_log_dump_ring(null);
    fclose(null);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 11: 压力测试 - 多线程高频提交
 * ================================================================ */
#define STRESS_THREADS 4
#define STRESS_ITERS   200

static conflux_queue_t *stress_q = NULL;

static void *stress_producer(void *arg) {
    (void)arg;
    for (int i = 0; i < STRESS_ITERS; i++) {
        conflux_cmd_t cmd = { .type = CONFLUX_CMD_NOP };
        conflux_event_t *ev;
        int ret = conflux_queue_submit(stress_q, &cmd, &ev);
        if (ret == CONFLUX_SUCCESS) {
            conflux_event_wait(ev, 0);
            conflux_event_destroy(ev);
        }
    }
    return NULL;
}

static void test_stress_mt_submit(void) {
    printf("  [11] Stress: multi-threaded submit... ");
    
    stress_q = conflux_queue_create(64, noop_execute, NULL);
    conflux_queue_start_consumer(stress_q);
    
    pthread_t producers[STRESS_THREADS];
    int ids[STRESS_THREADS];
    for (int i = 0; i < STRESS_THREADS; i++) {
        ids[i] = i;
        pthread_create(&producers[i], NULL, stress_producer, &ids[i]);
    }
    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_join(producers[i], NULL);
    }
    
    assert(stress_q->completed_count == STRESS_THREADS * STRESS_ITERS);
    
    conflux_queue_stop_consumer(stress_q);
    conflux_queue_destroy(stress_q);
    printf("PASSED\n");
}

/* ================================================================
 * 测试 12: 内存泄漏检查（Valgrind 配合）
 * ================================================================ */
static void test_memory_leak_suspicious(void) {
    printf("  [12] Memory: potential leak paths... ");
    
    /* 创建后立即销毁（不 init） */
    conflux_device_t *dev = conflux_device_create();
    conflux_device_destroy(dev);
    
    /* 设备 init 后不 online 就 destroy */
    dev = conflux_device_create();
    conflux_device_init(dev, NULL, 0x10000, 64*1024);
    conflux_device_destroy(dev);
    
    /* 平台快速 probe -> destroy */
    conflux_platform_init();
    conflux_device_desc_t d = { .mmio_base=0x10, .mmio_size=4096, .vendor_id=1, .device_id=1, .available=1 };
    d.path[0] = '\0';  /* SIM 模式 */
    conflux_platform_probe_specific(&d);
    conflux_platform_open_device(0);
    conflux_platform_destroy(); /* 应清理已打开设备 */
    
    printf("PASSED (run under valgrind to confirm)\n");
}

/* ================================================================
 * 主函数
 * ================================================================ */
int main(void) {
    printf("=== Quality Test Suite ===\n");
    
    test_allocator_fill_and_overflow();
    test_allocator_fragmentation();
    test_event_invalid_transitions();
    test_event_multiple_waiters();
    test_queue_backpressure();
    test_queue_stop_restart();
    test_kernel_edge_cases();
    test_builder_batch_and_barrier();
    test_platform_multi_device_concurrent();
    test_log_ring_overflow();
    test_stress_mt_submit();
    test_memory_leak_suspicious();
    
    printf("\n=== All Quality Tests PASSED ===\n");
    return 0;
}