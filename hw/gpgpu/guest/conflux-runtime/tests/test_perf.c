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
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

/* 测试规模 */
#define PERF_ITERATIONS     200000
#define WARMUP              1000
#define MT_ITERS_PER_THREAD 50000
#define MT_MAX_THREADS      4

/* 计时 */
static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* 结果收集 */
typedef struct {
    char   *name;   /* 动态分配，避免悬挂指针 */
    char   *unit;   /* 动态分配，保持一致 */
    double  value;
} perf_result_t;

#define MAX_RESULTS 20
static perf_result_t results[MAX_RESULTS];
static int result_count = 0;

static void add_result(const char *name, const char *unit, double value) {
    if (result_count < MAX_RESULTS) {
        results[result_count].name  = strdup(name);
        results[result_count].unit  = strdup(unit);
        results[result_count].value = value;
        result_count++;
    }
}

static void free_results(void) {
    for (int i = 0; i < result_count; i++) {
        free(results[i].name);
        free(results[i].unit);
    }
    result_count = 0;
}

/* ---- 1. 分配器 ---- */
static void perf_allocator(void) {
    conflux_allocator_t alloc;
    conflux_allocator_init(&alloc, 0, 64*1024*1024, 4096);

    uint64_t addrs[16384];
    for (int i = 0; i < 1000; i++) {
        addrs[i] = conflux_allocator_alloc(&alloc, 4096);
    }
    for (int i = 0; i < 1000; i++) {
        conflux_allocator_free(&alloc, addrs[i], 4096);
    }

    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        uint64_t a = conflux_allocator_alloc(&alloc, 4096);
        if (a != UINT64_MAX) conflux_allocator_free(&alloc, a, 4096);
    }
    uint64_t t1 = ns_now();
    double ops_per_sec = PERF_ITERATIONS / ((t1 - t0) / 1e9);
    add_result("allocator alloc+free", "ops/s", ops_per_sec);
    conflux_allocator_destroy(&alloc);
}

/* ---- 2. 事件创建/销毁 ---- */
static void perf_event_create_destroy(void) {
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        conflux_event_t *ev = conflux_event_create();
        conflux_event_destroy(ev);
    }
    uint64_t t1 = ns_now();
    double ops_per_sec = PERF_ITERATIONS / ((t1 - t0) / 1e9);
    add_result("event create+destroy", "ops/s", ops_per_sec);
}

/* ---- 3. 事件信号等待（热路径） ---- */
static void perf_event_signal_wait_hot(void) {
    conflux_event_t *ev = conflux_event_create();
    ev->status = CONFLUX_EVENT_SUBMITTED;
    uint64_t total_ns = 0;
    const int N = 20000;
    for (int i = 0; i < N; i++) {
        ev->status = CONFLUX_EVENT_SUBMITTED;
        uint64_t t0 = ns_now();
        conflux_event_set_complete(ev);
        conflux_event_wait(ev, 0);
        uint64_t t1 = ns_now();
        total_ns += (t1 - t0);
    }
    conflux_event_destroy(ev);
    double avg_ns = (double)total_ns / N;
    add_result("event signal+wait (hot)", "ns", avg_ns);
}

/* ---- 4. 队列吞吐 ---- */
static int noop_exec(conflux_cmd_t *cmd, void *ud) {
    (void)cmd; (void)ud;
    return 0;
}

static void perf_queue_throughput(void) {
    conflux_queue_t *q = conflux_queue_create(512, noop_exec, NULL);
    conflux_queue_start_consumer(q);
    conflux_cmd_t cmd = { .type = CONFLUX_CMD_NOP };
    for (int i = 0; i < WARMUP; i++) {
        conflux_event_t *ev;
        conflux_queue_submit(q, &cmd, &ev);
        conflux_event_wait(ev, 0);
        conflux_event_destroy(ev);
    }
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        conflux_event_t *ev;
        conflux_queue_submit(q, &cmd, &ev);
        conflux_event_wait(ev, 0);
        conflux_event_destroy(ev);
    }
    uint64_t t1 = ns_now();
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    double ops_per_sec = PERF_ITERATIONS / ((t1 - t0) / 1e9);
    add_result("queue submit+wait+free", "ops/s", ops_per_sec);
}

/* ---- 5. 内核打包 ---- */
static void perf_kernel_pack(void) {
    uint8_t dummy[256];
    conflux_kernel_t *k = conflux_kernel_create("perf", 1, dummy, sizeof(dummy));
    k->binary_device_addr = 0x1000;
    k->binary_uploaded = 1;
    conflux_kernel_set_arg(k, 0, 8, NULL, 0);
    conflux_kernel_set_arg(k, 1, 8, NULL, 0);
    conflux_kernel_set_global_size(k, 0, 256);
    conflux_cmd_t cmd;
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        conflux_kernel_pack_cmd(k, &cmd);
    }
    uint64_t t1 = ns_now();
    conflux_kernel_destroy(k);
    double ops_per_sec = PERF_ITERATIONS / ((t1 - t0) / 1e9);
    add_result("kernel pack_cmd", "ops/s", ops_per_sec);
}

/* ---- 6. 日志环写吞吐 ---- */
static void perf_log_ring(void) {
    conflux_log_init(CONFLUX_LOG_OFF);
    char buf[128]; memset(buf, 'A', 128);
    size_t total_bytes = 0;
    const int LOOPS = 50000;
    uint64_t t0 = ns_now();
    for (int i = 0; i < LOOPS; i++) {
        conflux_log_ring_write(&g_conflux_log.ring, (uint8_t*)buf, 100);
        total_bytes += 100;
    }
    uint64_t t1 = ns_now();
    double mb_sec = (total_bytes / (1024.0*1024.0)) / ((t1 - t0) / 1e9);
    add_result("log ring write 100B", "MB/s", mb_sec);
}

/* ---- 7. 多线程扩展 ---- */
typedef struct {
    conflux_queue_t *q;
    uint64_t        total_time;
    int             count;
} mt_worker_t;

static void *mt_worker(void *arg) {
    mt_worker_t *w = (mt_worker_t*)arg;
    conflux_cmd_t cmd = { .type = CONFLUX_CMD_NOP };
    uint64_t t0 = ns_now();
    for (int i = 0; i < MT_ITERS_PER_THREAD; i++) {
        conflux_event_t *ev;
        if (conflux_queue_submit(w->q, &cmd, &ev) == CONFLUX_SUCCESS) {
            conflux_event_wait(ev, 0);
            conflux_event_destroy(ev);
        }
    }
    w->total_time = ns_now() - t0;
    w->count = MT_ITERS_PER_THREAD;
    return NULL;
}

static void perf_mt_scalability(void) {
    for (int threads = 1; threads <= MT_MAX_THREADS; threads++) {
        conflux_queue_t *q = conflux_queue_create(512, noop_exec, NULL);
        conflux_queue_start_consumer(q);
        pthread_t tids[MT_MAX_THREADS];
        mt_worker_t workers[MT_MAX_THREADS];
        for (int i = 0; i < threads; i++) {
            workers[i].q = q;
            pthread_create(&tids[i], NULL, mt_worker, &workers[i]);
        }
        for (int i = 0; i < threads; i++) {
            pthread_join(tids[i], NULL);
        }
        uint64_t max_time = 0;
        int total_ops = 0;
        for (int i = 0; i < threads; i++) {
            if (workers[i].total_time > max_time) max_time = workers[i].total_time;
            total_ops += workers[i].count;
        }
        conflux_queue_stop_consumer(q);
        conflux_queue_destroy(q);
        double throughput = total_ops / (max_time / 1e9);
        char name[64];
        snprintf(name, sizeof(name), "queue %d threads", threads);
        add_result(name, "ops/s", throughput);
    }
}

/* ---- 总结表格 ---- */
static void print_summary(void) {
    printf("\n%70s\n", "==================== PERFORMANCE SUMMARY ===================");
    printf("%-32s %12s %18s\n", "Test", "Value", "Unit");
    printf("----------------------------------------------------------------\n");
    for (int i = 0; i < result_count; i++) {
        if (strcmp(results[i].unit, "ops/s") == 0) {
            printf("%-32s %12.2f %18s\n", results[i].name, results[i].value, results[i].unit);
        } else if (strcmp(results[i].unit, "MB/s") == 0) {
            printf("%-32s %12.2f %18s\n", results[i].name, results[i].value, results[i].unit);
        } else if (strcmp(results[i].unit, "ns") == 0) {
            printf("%-32s %12.1f %18s\n", results[i].name, results[i].value, results[i].unit);
        }
    }
    printf("----------------------------------------------------------------\n");

    printf("\nNotes:\n");
    printf(" - allocator: O(n) scan, ~16k blocks scanned per alloc on average.\n");
    printf(" - event signal+wait: hot path should be <200 ns, dominated by mutex+condvar.\n");
    printf(" - queue throughput: bounded by thread switches and consumer wakeup.\n");
    printf(" - kernel pack: essentially memcpy into cmd struct.\n");
    printf(" - log ring: lock-free ring buffer, very high throughput.\n");
    printf(" - multi-thread: check for near-linear scaling; if not, queue lock contention.\n");
}

int main(void) {
    printf("=== Running Performance Tests (Log OFF) ===\n");

    conflux_log_init(CONFLUX_LOG_OFF);

    perf_allocator();
    perf_event_create_destroy();
    perf_event_signal_wait_hot();
    perf_queue_throughput();
    perf_kernel_pack();
    perf_log_ring();
    perf_mt_scalability();
    
    print_summary();
    free_results();
    
    return 0;
}