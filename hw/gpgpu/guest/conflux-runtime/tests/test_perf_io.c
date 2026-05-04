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

#define PERF_ITERS 100000

static int noop_exec(conflux_cmd_t *cmd, void *ud) {
    (void)cmd; (void)ud; return 0;
}

/* ---- 计时 ---- */
static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ---- 单个测试项（返回 ops/sec） ---- */
static double test_allocator(void) {
    conflux_allocator_t alloc;
    conflux_allocator_init(&alloc, 0, 64*1024*1024, 4096);
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERS; i++) {
        uint64_t a = conflux_allocator_alloc(&alloc, 4096);
        if (a != UINT64_MAX) conflux_allocator_free(&alloc, a, 4096);
    }
    uint64_t t1 = ns_now();
    conflux_allocator_destroy(&alloc);
    return PERF_ITERS / ((t1 - t0) / 1e9);
}

static double test_event_cd(void) {
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERS; i++) {
        conflux_event_t *ev = conflux_event_create();
        conflux_event_destroy(ev);
    }
    uint64_t t1 = ns_now();
    return PERF_ITERS / ((t1 - t0) / 1e9);
}

static double test_event_signal_wait(void) {
    conflux_event_t *ev = conflux_event_create();
    ev->status = CONFLUX_EVENT_SUBMITTED;
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERS; i++) {
        ev->status = CONFLUX_EVENT_SUBMITTED;
        conflux_event_set_complete(ev);
        conflux_event_wait(ev, 0);
    }
    uint64_t t1 = ns_now();
    conflux_event_destroy(ev);
    return PERF_ITERS / ((t1 - t0) / 1e9);
}

static double test_queue(void) {
    conflux_queue_t *q = conflux_queue_create(512, noop_exec, NULL);
    conflux_queue_start_consumer(q);
    conflux_cmd_t cmd = { .type = CONFLUX_CMD_NOP };
    uint64_t t0 = ns_now();
    for (int i = 0; i < PERF_ITERS; i++) {
        conflux_event_t *ev;
        conflux_queue_submit(q, &cmd, &ev);
        conflux_event_wait(ev, 0);
        conflux_event_destroy(ev);
    }
    uint64_t t1 = ns_now();
    conflux_queue_stop_consumer(q);
    conflux_queue_destroy(q);
    return PERF_ITERS / ((t1 - t0) / 1e9);
}

/* ---- 主函数 ---- */
int main(void) {
    printf("=== I/O Overhead Comparison ===\n\n");

    double on_alloc, on_ev_cd, on_ev_sig, on_q;
    double off_alloc, off_ev_cd, off_ev_sig, off_q;

    // ===== 日志全开 =====
    conflux_log_init(CONFLUX_LOG_TRACE);   // 所有级别都输出
    g_conflux_log.timestamps = false;     // 减少时间戳格式化本身的开销
    printf("Running with FULL logging...\n");
    on_alloc   = test_allocator();
    on_ev_cd   = test_event_cd();
    on_ev_sig  = test_event_signal_wait();
    on_q       = test_queue();
    printf("Done.\n\n");

    // ===== 日志关闭 =====
    g_conflux_log.level = CONFLUX_LOG_OFF; // 所有日志宏什么都不做
    printf("Running with NO logging...\n");
    off_alloc  = test_allocator();
    off_ev_cd  = test_event_cd();
    off_ev_sig = test_event_signal_wait();
    off_q      = test_queue();
    printf("Done.\n\n");

    // ===== 输出表格 =====
    printf("%-30s %15s %15s %10s\n", "Test", "Log ON (op/s)", "Log OFF (op/s)", "Slowdown");
    printf("--------------------------------------------------------------------\n");
    #define SHOW(name, on, off) \
        printf("%-30s %15.0f %15.0f %9.1fx\n", name, on, off, off/on);
    SHOW("allocator alloc+free", on_alloc, off_alloc);
    SHOW("event create+destroy", on_ev_cd, off_ev_cd);
    SHOW("event signal+wait",    on_ev_sig, off_ev_sig);
    SHOW("queue submit+wait",   on_q, off_q);

    printf("\nVerdict:\n");
    printf("Logging overhead is significant for fine-grained operations.\n");
    printf("Always set CONFLUX_LOG_OFF before performance measurement.\n");
    return 0;
}