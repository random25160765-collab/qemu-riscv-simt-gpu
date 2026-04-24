/*
 * GPGPU 日志模块
 *
 * 三层架构：
 *   第一层（编译期）：GPGPU_LOG_ENABLED 决定模块是否存在
 *   第二层（运行时）：gpgpu_log_level / gpgpu_log_categories 决定是否记录
 *   第三层（运行时）：gpgpu_log_write_func 函数指针决定写往哪里
 */

#ifndef GPGPU_LOG_H
#define GPGPU_LOG_H

#include "qemu/osdep.h"

#if defined(__GNUC__) && !defined(__clang__)
#define GPGPU_LOG_PRINTF_ATTR __attribute__((__format__(__printf__, 5, 0)))
#else
#define GPGPU_LOG_PRINTF_ATTR
#endif

/* ============================================================
 * 第一层：编译期总开关
 * 设为 0 可彻底移除所有日志代码，零运行时开销
 * ============================================================ */
#define GPGPU_LOG_ENABLED 6

/* ============================================================
 * 第二层：日志级别枚举（详细程度从低到高）
 * ============================================================ */
typedef enum {
    GPGPU_LOG_OFF   = 0,
    GPGPU_LOG_ERROR = 1,
    GPGPU_LOG_INFO  = 2,
    GPGPU_LOG_DEV   = 3,
    GPGPU_LOG_CORE  = 4,
    GPGPU_LOG_INST  = 5,
    GPGPU_LOG_TRACE = 6,
} GPGPULogLevel;

/* 日志类别（位掩码，可组合） */
typedef enum {
    GPGPU_CAT_DEVICE = 1 << 0,   /* 设备级别事件（寄存器读写、初始化） */
    GPGPU_CAT_CORE   = 1 << 1,   /* 计算单元执行 */
    GPGPU_CAT_INST   = 1 << 2,   /* 指令级跟踪 */
    GPGPU_CAT_DMA    = 1 << 3,   /* DMA 传输 */
    GPGPU_CAT_INTR   = 1 << 4,   /* 中断处理 */
} GPGPULogCategory;

/* ============================================================
 * 第三层：输出路由函数指针类型
 * ============================================================ */
typedef void (*GPGPULogWriteFunc)(GPGPULogCategory cat, GPGPULogLevel level,
                                  const char *file, int line,
                                  const char *fmt, va_list args)
                                  GPGPU_LOG_PRINTF_ATTR;

/* ============================================================
 * 全局状态（定义在 gpgpu_log.c）
 * ============================================================ */
extern GPGPULogLevel    gpgpu_log_level;
extern uint32_t         gpgpu_log_categories;
extern GPGPULogWriteFunc gpgpu_log_write_func;

/* ============================================================
 * API
 * ============================================================ */
void gpgpu_log_init(void);

/* 运行时切换级别/类别 */
static inline void gpgpu_log_set_level(GPGPULogLevel level)
{
    gpgpu_log_level = level;
}

static inline void gpgpu_log_set_categories(uint32_t cats)
{
    gpgpu_log_categories = cats;
}

/* 运行时切换输出路由："null" / "stdio" / "ringbuf" */
void gpgpu_log_set_output(const char *output);

/* 核心写入函数（不应直接调用，通过宏调用） */
void gpgpu_log_write(GPGPULogCategory cat, GPGPULogLevel level,
                     const char *file, int line,
                     const char *fmt, ...) G_GNUC_PRINTF(5, 6);

/* ============================================================
 * 日志宏
 * ============================================================ */
#if GPGPU_LOG_ENABLED

#define GPGPU_LOG(cat, level, fmt, ...)                                  \
    do {                                                                  \
        if (gpgpu_log_write_func != NULL &&                              \
            (level) <= gpgpu_log_level &&                                \
            ((cat) & gpgpu_log_categories)) {                            \
            gpgpu_log_write(cat, level, __FILE__, __LINE__,              \
                            fmt, ##__VA_ARGS__);                          \
        }                                                                 \
    } while (0)

#else  /* 编译期关闭：彻底消除，零开销 */

#define GPGPU_LOG(cat, level, fmt, ...) ((void)0)

#endif /* GPGPU_LOG_ENABLED */

/* 便捷宏 */
#define GPGPU_ERR(fmt, ...)  GPGPU_LOG(GPGPU_CAT_DEVICE, GPGPU_LOG_ERROR, fmt, ##__VA_ARGS__)
#define GPGPU_INFO(fmt, ...) GPGPU_LOG(GPGPU_CAT_DEVICE, GPGPU_LOG_INFO,  fmt, ##__VA_ARGS__)
#define GPGPU_DEV(fmt, ...)  GPGPU_LOG(GPGPU_CAT_DEVICE, GPGPU_LOG_DEV,   fmt, ##__VA_ARGS__)
#define GPGPU_CORE(fmt, ...) GPGPU_LOG(GPGPU_CAT_CORE,   GPGPU_LOG_CORE,  fmt, ##__VA_ARGS__)
#define GPGPU_INST(fmt, ...) GPGPU_LOG(GPGPU_CAT_INST,   GPGPU_LOG_INST,  fmt, ##__VA_ARGS__)
#define GPGPU_DMA(fmt, ...)  GPGPU_LOG(GPGPU_CAT_DMA,    GPGPU_LOG_DEV,   fmt, ##__VA_ARGS__)
#define GPGPU_INTR(fmt, ...) GPGPU_LOG(GPGPU_CAT_INTR,   GPGPU_LOG_DEV,   fmt, ##__VA_ARGS__)

#endif /* GPGPU_LOG_H */
