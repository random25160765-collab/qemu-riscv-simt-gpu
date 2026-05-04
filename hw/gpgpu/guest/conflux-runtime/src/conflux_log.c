#include "conflux_log.h"
#include <stdarg.h>
#include <time.h>

/* ================================================================
 * 环形缓冲区
 * ================================================================ */

void conflux_log_ring_init(conflux_log_ring_t *rb) {
    atomic_init(&rb->r, 0);
    atomic_init(&rb->w, 0);
}

int conflux_log_ring_peek(conflux_log_ring_t *rb, struct iovec iov[2]) {
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    size_t available = w - r;
    if (available == 0) return 0;
    if (available > CONFLUX_LOG_RING_SIZE) available = CONFLUX_LOG_RING_SIZE;

    uint32_t r_phy = r & (CONFLUX_LOG_RING_SIZE - 1);
    size_t first_len = CONFLUX_LOG_RING_SIZE - r_phy;

    if (available <= first_len) {
        iov[0].iov_base = &rb->buf[r_phy];
        iov[0].iov_len = available;
        return 1;
    } else {
        iov[0].iov_base = &rb->buf[r_phy];
        iov[0].iov_len = first_len;
        iov[1].iov_base = &rb->buf[0];
        iov[1].iov_len = available - first_len;
        return 2;
    }
}

void conflux_log_ring_commit(conflux_log_ring_t *rb, size_t len) {
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);
    atomic_store_explicit(&rb->r, r + len, memory_order_release);
}

int conflux_log_ring_write(conflux_log_ring_t *rb, const uint8_t *src, size_t len) {
    if (len == 0) return 0;
    if (len > CONFLUX_LOG_RING_SIZE) {
        src += len - CONFLUX_LOG_RING_SIZE;
        len = CONFLUX_LOG_RING_SIZE;
    }

    uint32_t r = atomic_load_explicit(&rb->r, memory_order_acquire);
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_relaxed);

    size_t available = CONFLUX_LOG_RING_SIZE - (w - r);
    if (available < len) {
        size_t new_r = r + len - available;
        atomic_store_explicit(&rb->r, new_r, memory_order_release);
    }

    uint32_t w_phy = w & (CONFLUX_LOG_RING_SIZE - 1);
    size_t first_write = CONFLUX_LOG_RING_SIZE - w_phy;
    if (len <= first_write) {
        memcpy(&rb->buf[w_phy], src, len);
    } else {
        memcpy(&rb->buf[w_phy], src, first_write);
        memcpy(&rb->buf[0], src + first_write, len - first_write);
    }
    atomic_store_explicit(&rb->w, w + len, memory_order_release);

    return 0;
}

size_t conflux_log_ring_read(conflux_log_ring_t *rb, uint8_t *dst, size_t len) {
    if (len == 0) return 0;
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    size_t available = w - r;
    if (available < len) len = available;

    uint32_t r_phy = r & (CONFLUX_LOG_RING_SIZE - 1);
    size_t first_read = CONFLUX_LOG_RING_SIZE - r_phy;
    if (len <= first_read) {
        memcpy(dst, &rb->buf[r_phy], len);
    } else {
        memcpy(dst, &rb->buf[r_phy], first_read);
        memcpy(dst + first_read, &rb->buf[0], len - first_read);
    }
    atomic_store_explicit(&rb->r, r + len, memory_order_release);

    return len;
}

/* ================================================================
 * 日志系统
 * ================================================================ */

/* 全局实例 */
conflux_log_t g_conflux_log = {0};

/* 日志级别对应的颜色和前缀 */
static const char *level_color(conflux_log_level_t level) {
    switch (level) {
        case CONFLUX_LOG_TRACE: return "\033[90m";    /* 灰色 */
        case CONFLUX_LOG_DEBUG: return "\033[36m";    /* 青色 */
        case CONFLUX_LOG_INFO:  return "\033[32m";    /* 绿色 */
        case CONFLUX_LOG_WARN:  return "\033[33m";    /* 黄色 */
        case CONFLUX_LOG_ERROR: return "\033[31m";    /* 红色 */
        case CONFLUX_LOG_FATAL: return "\033[35m";    /* 紫色 */
        default:               return "\033[0m";
    }
}

static const char *level_str(conflux_log_level_t level) {
    switch (level) {
        case CONFLUX_LOG_TRACE: return "TRACE";
        case CONFLUX_LOG_DEBUG: return "DEBUG";
        case CONFLUX_LOG_INFO:  return "INFO ";
        case CONFLUX_LOG_WARN:  return "WARN ";
        case CONFLUX_LOG_ERROR: return "ERROR";
        case CONFLUX_LOG_FATAL: return "FATAL";
        default:               return "?????";
    }
}

void conflux_log_init(conflux_log_level_t level) {
    conflux_log_ring_init(&g_conflux_log.ring);
    g_conflux_log.level      = level;
    g_conflux_log.timestamps = true;
    g_conflux_log.colors     = true;
    pthread_mutex_init(&g_conflux_log.stdout_lock, NULL);
    
    CONFLUX_INFO("Log system initialized (level=%d)", level);
}

void conflux_log_write(conflux_log_level_t level,
                      const char *file, int line,
                      const char *fmt, ...) 
{
    /* 级别过滤 */
    if (level < g_conflux_log.level) return;
    
    char msg[256];
    char buf[512];
    int msg_len;
    
    /* 格式化用户消息 */
    va_list ap;
    va_start(ap, fmt);
    msg_len = vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (msg_len < 0) return;
    if ((size_t)msg_len >= sizeof(msg)) msg_len = sizeof(msg) - 1;
    
    /* 构建完整日志行 */
    int total;
    if (g_conflux_log.timestamps) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        struct tm tm;
        localtime_r(&ts.tv_sec, &tm);
        
        total = snprintf(buf, sizeof(buf),
                         "%02d:%02d:%02d.%03ld [%s] %s:%d: %s\n",
                         tm.tm_hour, tm.tm_min, tm.tm_sec,
                         ts.tv_nsec / 1000000,
                         level_str(level),
                         file, line,
                         msg);
    } else {
        total = snprintf(buf, sizeof(buf),
                         "[%s] %s:%d: %s\n",
                         level_str(level), file, line, msg);
    }
    if (total <= 0) return;
    
    /* 写入环形缓冲区（崩溃后能 dump） */
    conflux_log_ring_write(&g_conflux_log.ring, (uint8_t *)buf, total);
    
    /* 同步写入 stderr（带颜色） */
    pthread_mutex_lock(&g_conflux_log.stdout_lock);
    if (g_conflux_log.colors) {
        fprintf(stderr, "%s%s\033[0m", level_color(level), buf);
    } else {
        fprintf(stderr, "%s", buf);
    }
    fflush(stderr);
    pthread_mutex_unlock(&g_conflux_log.stdout_lock);
    
    /* FATAL 级别直接 abort */
    if (level == CONFLUX_LOG_FATAL) {
        fprintf(stderr, "FATAL error, aborting.\n");
        conflux_log_dump_ring(stderr);
        abort();
    }
}

void conflux_log_dump_ring(FILE *out) {
    if (!out) out = stderr;
    
    fprintf(out, "\n=== Log Ring Buffer Dump ===\n");
    
    struct iovec iov[2];
    int n = conflux_log_ring_peek(&g_conflux_log.ring, iov);
    
    for (int i = 0; i < n; i++) {
        fwrite(iov[i].iov_base, 1, iov[i].iov_len, out);
    }
    
    fprintf(out, "=== End of Log Dump ===\n\n");
    fflush(out);
}

/* ================================================================
 * 性能统计
 * ================================================================ */

conflux_perf_stats_t g_conflux_perf = {0};

void conflux_perf_init(void) {
    memset(&g_conflux_perf, 0, sizeof(g_conflux_perf));
    g_conflux_perf.min_wait_ns = UINT64_MAX;
}

void conflux_perf_record_submit(uint32_t cmd_type) {
    g_conflux_perf.total_submits++;
    switch (cmd_type) {
        case 1: g_conflux_perf.total_copies++;   break;  /* CMD_COPY */
        case 2: g_conflux_perf.total_kernels++;  break;  /* CMD_KERNEL */
        case 3: g_conflux_perf.total_allocs++;   break;  /* CMD_ALLOC */
        case 4: g_conflux_perf.total_frees++;    break;  /* CMD_FREE */
        case 5: g_conflux_perf.total_barriers++; break;  /* CMD_BARRIER */
        default: break;
    }
}

void conflux_perf_record_copy(size_t bytes) {
    g_conflux_perf.total_bytes_copied += bytes;
}

void conflux_perf_record_alloc(size_t bytes) {
    g_conflux_perf.total_bytes_alloced += bytes;
}

void conflux_perf_record_wait(uint64_t wait_ns) {
    conflux_perf_stats_t *s = &g_conflux_perf;
    
    if (wait_ns < s->min_wait_ns) s->min_wait_ns = wait_ns;
    if (wait_ns > s->max_wait_ns) s->max_wait_ns = wait_ns;
    s->total_wait_ns += wait_ns;
    s->wait_samples++;
    s->avg_wait_ns = s->total_wait_ns / s->wait_samples;
}

void conflux_perf_record_error(void) {
    g_conflux_perf.total_errors++;
}

void conflux_perf_record_timeout(void) {
    g_conflux_perf.total_timeouts++;
}

void conflux_perf_dump(const conflux_perf_stats_t *s, char *buf, size_t buf_size) {
    if (!s || !buf) return;
    
    snprintf(buf, buf_size,
             "=== Performance Stats ===\n"
             "  Commands:\n"
             "    total submits:  %lu\n"
             "    copies:         %lu\n"
             "    kernels:        %lu\n"
             "    allocs:         %lu\n"
             "    frees:          %lu\n"
             "    barriers:       %lu\n"
             "  Data:\n"
             "    bytes copied:   %lu\n"
             "    bytes alloced:  %lu\n"
             "  Timing (ns):\n"
             "    samples:        %lu\n"
             "    avg wait:       %lu\n"
             "    min wait:       %lu\n"
             "    max wait:       %lu\n"
             "  Errors:\n"
             "    total errors:   %lu\n"
             "    timeouts:       %lu\n",
             (unsigned long)s->total_submits,
             (unsigned long)s->total_copies,
             (unsigned long)s->total_kernels,
             (unsigned long)s->total_allocs,
             (unsigned long)s->total_frees,
             (unsigned long)s->total_barriers,
             (unsigned long)s->total_bytes_copied,
             (unsigned long)s->total_bytes_alloced,
             (unsigned long)s->wait_samples,
             (unsigned long)s->avg_wait_ns,
             (unsigned long)s->min_wait_ns,
             (unsigned long)s->max_wait_ns,
             (unsigned long)s->total_errors,
             (unsigned long)s->total_timeouts);
}

void conflux_perf_print(void) {
    char buf[1024];
    conflux_perf_dump(&g_conflux_perf, buf, sizeof(buf));
    fprintf(stderr, "%s\n", buf);
}