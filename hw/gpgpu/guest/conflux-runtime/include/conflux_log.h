#ifndef CONFLUX_LOG_H
#define CONFLUX_LOG_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <sys/uio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

/* ================================================================
 * 环形缓冲区
 * ================================================================ */

#define CONFLUX_LOG_RING_SIZE (4 * 1024)

#if (CONFLUX_LOG_RING_SIZE & (CONFLUX_LOG_RING_SIZE - 1)) != 0
#error "CONFLUX_LOG_RING_SIZE must be a power of 2"
#endif

typedef struct {
    uint8_t buf[CONFLUX_LOG_RING_SIZE];
    _Atomic uint32_t r;
    _Atomic uint32_t w;
} conflux_log_ring_t;

void conflux_log_ring_init(conflux_log_ring_t *rb);
int  conflux_log_ring_peek(conflux_log_ring_t *rb, struct iovec iov[2]);
void conflux_log_ring_commit(conflux_log_ring_t *rb, size_t len);
int  conflux_log_ring_write(conflux_log_ring_t *rb, const uint8_t *src, size_t len);
size_t conflux_log_ring_read(conflux_log_ring_t *rb, uint8_t *dst, size_t len);

/* ================================================================
 * 日志级别
 * ================================================================ */

typedef enum {
    CONFLUX_LOG_TRACE = 0,   /* 最详细 */
    CONFLUX_LOG_DEBUG = 1,
    CONFLUX_LOG_INFO  = 2,
    CONFLUX_LOG_WARN  = 3,
    CONFLUX_LOG_ERROR = 4,
    CONFLUX_LOG_FATAL = 5,
    CONFLUX_LOG_OFF   = 6,
} conflux_log_level_t;

/* ================================================================
 * 日志系统
 * ================================================================ */

typedef struct {
    conflux_log_ring_t ring;              /* 无锁环形缓冲区 */
    conflux_log_level_t level;            /* 当前日志级别 */
    pthread_mutex_t    stdout_lock;      /* 保护 stdout 输出 */
    bool               timestamps;       /* 是否加时间戳 */
    bool               colors;           /* 是否用 ANSI 颜色 */
} conflux_log_t;

/* 全局日志实例 */
extern conflux_log_t g_conflux_log;

/* 初始化 */
void conflux_log_init(conflux_log_level_t level);

/* 格式化写入 */
void conflux_log_write(conflux_log_level_t level, 
                      const char *file, int line,
                      const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));

/* 便捷宏 */
#define CONFLUX_TRACE(fmt, ...) \
    conflux_log_write(CONFLUX_LOG_TRACE, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define CONFLUX_DEBUG(fmt, ...) \
    conflux_log_write(CONFLUX_LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define CONFLUX_INFO(fmt, ...)  \
    conflux_log_write(CONFLUX_LOG_INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define CONFLUX_WARN(fmt, ...)  \
    conflux_log_write(CONFLUX_LOG_WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define CONFLUX_ERROR(fmt, ...) \
    conflux_log_write(CONFLUX_LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define CONFLUX_FATAL(fmt, ...) \
    conflux_log_write(CONFLUX_LOG_FATAL, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/* 崩溃时 dump 日志环到文件/stderr */
void conflux_log_dump_ring(FILE *out);

/* ================================================================
 * 性能统计
 * ================================================================ */

typedef struct {
    /* 命令统计 */
    uint64_t total_submits;
    uint64_t total_copies;
    uint64_t total_kernels;
    uint64_t total_allocs;
    uint64_t total_frees;
    uint64_t total_barriers;
    
    /* 数据量统计 */
    uint64_t total_bytes_copied;
    uint64_t total_bytes_alloced;
    
    /* 时间统计（纳秒） */
    uint64_t min_wait_ns;
    uint64_t max_wait_ns;
    uint64_t avg_wait_ns;
    uint64_t total_wait_ns;
    uint64_t wait_samples;
    
    /* 错误统计 */
    uint64_t total_errors;
    uint64_t total_timeouts;
    
} conflux_perf_stats_t;

/* 全局性能统计 */
extern conflux_perf_stats_t g_conflux_perf;

/* 初始化 */
void conflux_perf_init(void);

/* 记录事件 */
void conflux_perf_record_submit(uint32_t cmd_type);
void conflux_perf_record_copy(size_t bytes);
void conflux_perf_record_alloc(size_t bytes);
void conflux_perf_record_wait(uint64_t wait_ns);
void conflux_perf_record_error(void);
void conflux_perf_record_timeout(void);

/* 查询 */
void conflux_perf_dump(const conflux_perf_stats_t *stats, char *buf, size_t buf_size);
void conflux_perf_print(void);

#endif