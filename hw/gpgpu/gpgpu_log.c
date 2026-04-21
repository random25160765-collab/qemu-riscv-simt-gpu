/*
 * GPGPU 日志模块实现
 *
 * 三层架构实现：
 *   第一层：编译期开关（见 gpgpu_log.h 中 GPGPU_LOG_ENABLED）
 *   第二层：运行时级别/类别过滤（gpgpu_log_level / gpgpu_log_categories）
 *   第三层：运行时输出路由（gpgpu_log_write_func 函数指针）
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu_log.h"

/* ============================================================
 * 全局状态
 * ============================================================ */
GPGPULogLevel     gpgpu_log_level      = GPGPU_LOG_INFO;   /* 默认 INFO */
uint32_t          gpgpu_log_categories = 0xFFFFFFFF;        /* 默认全类别开启 */
GPGPULogWriteFunc gpgpu_log_write_func = NULL;              /* 初始化前为空 */

/* ============================================================
 * 第三层：各输出路由实现
 * ============================================================ */

/* 路由一：空路由，什么都不做 */
static void log_write_null(GPGPULogCategory cat, GPGPULogLevel level,
                           const char *file, int line,
                           const char *fmt, va_list args)
{
    (void)cat; (void)level; (void)file; (void)line; (void)fmt; (void)args;
}

/* 路由二：直接输出到 qemu_log（stderr / -D 文件）
 * 消息本身已含 [DEVICE] / \t[KERNEL] / \t\t[INST] 层级标记，
 * 此处不额外添加前缀，保持三级制表缩进层次可读。 */
static void log_write_stdio(GPGPULogCategory cat, GPGPULogLevel level,
                            const char *file, int line,
                            const char *fmt, va_list args)
{
    (void)cat; (void)level; (void)file; (void)line;

    va_list args2;
    va_copy(args2, args);
    qemu_log(fmt, args2);
    va_end(args2);
}

/* 路由三：环形缓冲区（内存中循环写入，用于高频日志不阻塞执行） */
#define RINGBUF_SIZE (1 * 1024 * 1024)  /* 1MB */
static char  ringbuf[RINGBUF_SIZE];
static int   ringbuf_head = 0;

static void GPGPU_LOG_PRINTF_ATTR
log_write_ringbuf(GPGPULogCategory cat, GPGPULogLevel level,
                  const char *file, int line,
                  const char *fmt, va_list args)
{
    char tmp[512];
    int  len;

    (void)cat; (void)level; (void)file; (void)line;

    len = vsnprintf(tmp, sizeof(tmp), fmt, args);
    if (len <= 0) {
        return;
    }
    if (len >= (int)sizeof(tmp)) {
        len = sizeof(tmp) - 1;
    }

    for (int i = 0; i < len; i++) {
        ringbuf[ringbuf_head] = tmp[i];
        ringbuf_head = (ringbuf_head + 1) % RINGBUF_SIZE;
    }
}

/* ============================================================
 * 运行时切换输出路由
 * ============================================================ */
void gpgpu_log_set_output(const char *output)
{
    if (strcmp(output, "null") == 0) {
        gpgpu_log_write_func = log_write_null;
    } else if (strcmp(output, "stdio") == 0) {
        gpgpu_log_write_func = log_write_stdio;
    } else if (strcmp(output, "ringbuf") == 0) {
        gpgpu_log_write_func = log_write_ringbuf;
    }
    /* 未知名称时保持原有路由不变 */
}

/* ============================================================
 * 初始化：默认输出到 stdio
 * ============================================================ */
void gpgpu_log_init(void)
{
    gpgpu_log_write_func = log_write_stdio;
}

/* ============================================================
 * 核心写入函数（由宏 GPGPU_LOG 调用）
 * 此处完成二层过滤已在宏内完成，直接调用函数指针
 * ============================================================ */
void gpgpu_log_write(GPGPULogCategory cat, GPGPULogLevel level,
                     const char *file, int line,
                     const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    gpgpu_log_write_func(cat, level, file, line, fmt, args);
    va_end(args);
}
