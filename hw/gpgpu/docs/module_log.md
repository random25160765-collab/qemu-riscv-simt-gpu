# 日志模块

## 一、分工边界

| 维度 | 编译期开关 (`#ifdef`) | 运行时函数指针 |
|:---|:---|:---|
| **负责问题** | 代码是否存在 | 代码执行哪条路径 |
| **控制粒度** | 整个日志模块的生死 | 日志输出的目标（环形缓冲/stdio/空） |
| **开销层级** | 零开销（代码都不编译进去） | 一次函数指针调用 + 几次条件判断 |
| **切换成本** | 重新编译 | 运行时原子赋值，纳秒级 |
| **典型场景** | 生产版本彻底移除日志 | 开发版本动态调节输出目标 |

**类比**：
- 编译期开关 = 盖楼时决定要不要电梯井
- 运行时函数指针 = 电梯装好后，按几楼

---

## 二、协调机制：三层架构

```
┌─────────────────────────────────────────────────────────────┐
│ 第一层：编译期开关（决定模块是否存在）                         │
│                                                             │
│   #if GPGPU_LOG_ENABLED                                     │
│       // 整个日志框架的代码                                   │
│   #else                                                     │
│       #define GPGPU_LOG(...) ((void)0)  // 彻底抹掉          │
│   #endif                                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 第二层：运行时类别/级别过滤（决定要不要记录）                   │
│                                                             │
│   if (level > current_log_level) return;                    │
│   if (!(category & enabled_categories)) return;             │
│                                                             │
│   控制方式：全局变量，运行时通过 Monitor 修改                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 第三层：运行时输出路由（决定写到哪）                           │
│                                                             │
│   log_write_func(level, category, fmt, args);               │
│                                                             │
│   可选实现：                                                  │
│   - log_write_null     → 什么都不做                          │
│   - log_write_ringbuf  → 写入环形缓冲区                      │
│   - log_write_stdio    → 直接 qemu_log                       │
│   - log_write_trace    → 二进制 trace buffer                │
│                                                             │
│   控制方式：运行时替换函数指针                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、三级日志控制的具体实现

你说的"三个级别的日志"，我理解是指三种不同粒度的控制需求：

| 级别 | 控制内容 | 实现方式 |
|:---|:---|:---|
| **级别一：生死开关** | 日志模块是否存在 | 编译期 `#ifdef` |
| **级别二：详细程度** | ERROR / INFO / DEV / CORE / INST / TRACE | 运行时 `current_log_level` 变量 |
| **级别三：输出路由** | 写到环形缓冲区 / 直接打印 / 丢弃 | 运行时 `log_write_func` 函数指针 |

### 完整代码实现

```c
// ==================== gpgpu_config.h ====================
// 编译期总开关
#define GPGPU_LOG_COMPILE_TIME  1   // 1=编译日志模块, 0=彻底移除

// ==================== gpgpu_log.h ====================
#ifndef GPGPU_LOG_H
#define GPGPU_LOG_H

#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>

// ---------- 级别二：详细程度 ----------
typedef enum {
    LOG_OFF   = 0,
    LOG_ERROR = 1,
    LOG_INFO  = 2,
    LOG_DEV   = 3,
    LOG_CORE  = 4,
    LOG_INST  = 5,
    LOG_TRACE = 6,
} GPGPULogLevel;

// 类别（位掩码）
typedef enum {
    LOG_CAT_DEVICE  = 1 << 0,
    LOG_CAT_CORE    = 1 << 1,
    LOG_CAT_INST    = 1 << 2,
    LOG_CAT_DMA     = 1 << 3,
    LOG_CAT_INTR    = 1 << 4,
} GPGPULogCategory;

// ---------- 级别三：输出路由函数指针 ----------
typedef void (*LogWriteFunc)(GPGPULogCategory cat, GPGPULogLevel level,
                             const char* file, int line,
                             const char* fmt, va_list args);

// ---------- 全局状态 ----------
extern GPGPULogLevel gpgpu_log_level;
extern uint32_t gpgpu_log_categories;
extern LogWriteFunc gpgpu_log_write_func;

// ---------- API ----------
void gpgpu_log_init(void);

// 级别二控制：详细程度
static inline void gpgpu_log_set_level(GPGPULogLevel level) {
    gpgpu_log_level = level;
}
static inline void gpgpu_log_set_categories(uint32_t cats) {
    gpgpu_log_categories = cats;
}

// 级别三控制：输出路由
void gpgpu_log_set_output(const char* output);

// 核心日志函数
void _gpgpu_log_write(GPGPULogCategory cat, GPGPULogLevel level,
                      const char* file, int line,
                      const char* fmt, ...);

#endif
```

```c
// ==================== gpgpu_log.c ====================
#include "gpgpu_log.h"

// 全局状态
GPGPULogLevel gpgpu_log_level = LOG_INFO;      // 默认 INFO
uint32_t gpgpu_log_categories = 0xFFFFFFFF;     // 默认全开
LogWriteFunc gpgpu_log_write_func = NULL;

// ---------- 输出路由实现 ----------
static void log_write_null(GPGPULogCategory cat, GPGPULogLevel level,
                           const char* file, int line,
                           const char* fmt, va_list args) {
    // 什么都不做
}

static void log_write_stdio(GPGPULogCategory cat, GPGPULogLevel level,
                            const char* file, int line,
                            const char* fmt, va_list args) {
    const char* level_str[] = {"OFF", "ERR", "INFO", "DEV", "CORE", "INST", "TRACE"};
    fprintf(stderr, "[%s] ", level_str[level]);
    if (level >= LOG_DEV) {
        fprintf(stderr, "%s:%d: ", file, line);
    }
    vfprintf(stderr, fmt, args);
}

static char ringbuf[1024 * 1024];  // 1MB
static int ringbuf_head = 0;

static void log_write_ringbuf(GPGPULogCategory cat, GPGPULogLevel level,
                              const char* file, int line,
                              const char* fmt, va_list args) {
    char tmp[256];
    int len = vsnprintf(tmp, sizeof(tmp), fmt, args);
    for (int i = 0; i < len; i++) {
        ringbuf[ringbuf_head] = tmp[i];
        ringbuf_head = (ringbuf_head + 1) % sizeof(ringbuf);
    }
}

// 运行时切换输出路由
void gpgpu_log_set_output(const char* output) {
    if (strcmp(output, "null") == 0) {
        gpgpu_log_write_func = log_write_null;
    } else if (strcmp(output, "stdio") == 0) {
        gpgpu_log_write_func = log_write_stdio;
    } else if (strcmp(output, "ringbuf") == 0) {
        gpgpu_log_write_func = log_write_ringbuf;
    }
}

void gpgpu_log_init(void) {
    gpgpu_log_write_func = log_write_ringbuf;  // 默认环形缓冲
}

// 核心写入函数
void _gpgpu_log_write(GPGPULogCategory cat, GPGPULogLevel level,
                      const char* file, int line,
                      const char* fmt, ...) {
    if (gpgpu_log_write_func == NULL) return;
    
    va_list args;
    va_start(args, fmt);
    gpgpu_log_write_func(cat, level, file, line, fmt, args);
    va_end(args);
}
```

```c
// ==================== gpgpu_log_inline.h ====================
// 日志宏（用户只 include 这个）

#if GPGPU_LOG_COMPILE_TIME

#define GPGPU_LOG(cat, level, fmt, ...) \
    do { \
        if (gpgpu_log_write_func != NULL && \
            (level) <= gpgpu_log_level && \
            (cat) & gpgpu_log_categories) { \
            _gpgpu_log_write(cat, level, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#else  // 编译期关闭

#define GPGPU_LOG(cat, level, fmt, ...) ((void)0)

#endif

// 便捷宏
#define LOG_ERROR(fmt, ...)  GPGPU_LOG(LOG_CAT_DEVICE, LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)   GPGPU_LOG(LOG_CAT_DEVICE, LOG_INFO,  fmt, ##__VA_ARGS__)
#define LOG_DEV(fmt, ...)    GPGPU_LOG(LOG_CAT_DEVICE, LOG_DEV,   fmt, ##__VA_ARGS__)
#define LOG_CORE(fmt, ...)   GPGPU_LOG(LOG_CAT_CORE,   LOG_CORE,  fmt, ##__VA_ARGS__)
#define LOG_INST(fmt, ...)   GPGPU_LOG(LOG_CAT_INST,   LOG_INST,  fmt, ##__VA_ARGS__)
```

---

## 四、运行时控制（QEMU Monitor）

```c
// 注册 Monitor 命令
void hmp_gpgpu_log(Monitor* mon, const QDict* qdict) {
    const char* level_str = qdict_get_try_str(qdict, "level");
    const char* output_str = qdict_get_try_str(qdict, "output");
    const char* cat_str = qdict_get_try_str(qdict, "category");
    
    if (level_str) {
        if (strcmp(level_str, "off") == 0)   gpgpu_log_level = LOG_OFF;
        if (strcmp(level_str, "info") == 0)  gpgpu_log_level = LOG_INFO;
        if (strcmp(level_str, "dev") == 0)   gpgpu_log_level = LOG_DEV;
        if (strcmp(level_str, "core") == 0)  gpgpu_log_level = LOG_CORE;
        if (strcmp(level_str, "inst") == 0)  gpgpu_log_level = LOG_INST;
        monitor_printf(mon, "Log level set to: %s\n", level_str);
    }
    
    if (output_str) {
        gpgpu_log_set_output(output_str);
        monitor_printf(mon, "Log output set to: %s\n", output_str);
    }
    
    if (cat_str) {
        if (strcmp(cat_str, "all") == 0)   gpgpu_log_categories = 0xFFFFFFFF;
        if (strcmp(cat_str, "core") == 0)  gpgpu_log_categories = LOG_CAT_CORE;
        if (strcmp(cat_str, "inst") == 0)  gpgpu_log_categories = LOG_CAT_INST;
        monitor_printf(mon, "Log categories set to: %s\n", cat_str);
    }
}
```

使用示例：
```bash
(qemu) gpgpu-log level=inst output=ringbuf category=core
(qemu) gpgpu-log level=off
(qemu) gpgpu-log output=stdio level=dev
```

---

## 五、总结：三层职责一览

| 层次 | 控制方式 | 切换成本 | 控制内容 |
|:---|:---|:---|:---|
| **第一层：编译期** | `#ifdef GPGPU_LOG_COMPILE_TIME` | 重新编译 | 日志模块是否存在 |
| **第二层：运行时级别** | 修改全局变量 `gpgpu_log_level` | 纳秒级 | ERROR/INFO/DEV/CORE/INST |
| **第三层：输出路由** | 替换函数指针 `gpgpu_log_write_func` | 纳秒级 | null / stdio / ringbuf / trace |

**一句话总结**：
> **编译期决定"有没有"，运行时级别决定"记不记"，函数指针决定"往哪记"。三层正交，互不干扰。**