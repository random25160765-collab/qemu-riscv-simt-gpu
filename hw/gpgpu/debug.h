#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <stdint.h>
#include "utils.h"

/* ========== 调试开关 ========== */
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 4  // 0: 无调试, 1: 错误, 2: 警告, 3: 信息, 4: 详细
#endif

/* ========== 日志级别 ========== */
#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_INFO  3
#define LOG_LEVEL_DEBUG 4
#define LOG_LEVEL_TRACE 5

/* ========== 基础日志宏 ========== */
#define Log(level, color, prefix, format, ...) \
    do { \
        if (DEBUG_LEVEL >= level) { \
            printf(color "[" prefix "] %s:%d %s(): " format COLOR_RESET "\n", \
                   __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

/* ========== 各级别日志 ========== */
#define LogError(format, ...)   Log(LOG_LEVEL_ERROR, COLOR_RED,    "ERROR", format, ## __VA_ARGS__)
#define LogWarn(format, ...)    Log(LOG_LEVEL_WARN,  COLOR_YELLOW, "WARN",  format, ## __VA_ARGS__)
#define LogInfo(format, ...)    Log(LOG_LEVEL_INFO,  COLOR_GREEN,  "INFO",  format, ## __VA_ARGS__)
#define LogDebug(format, ...)   Log(LOG_LEVEL_DEBUG, COLOR_CYAN,   "DEBUG", format, ## __VA_ARGS__)
#define LogTrace(format, ...)   Log(LOG_LEVEL_TRACE, COLOR_WHITE,  "TRACE", format, ## __VA_ARGS__)

/* ========== 简化日志 (不带文件/行号) ========== */
#define LogSimple(level, color, format, ...) \
    do { \
        if (DEBUG_LEVEL >= level) { \
            printf(color format COLOR_RESET "\n", ## __VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

#define Info(format, ...)  LogSimple(LOG_LEVEL_INFO,  COLOR_GREEN,  format, ## __VA_ARGS__)
#define Warn(format, ...)  LogSimple(LOG_LEVEL_WARN,  COLOR_YELLOW, format, ## __VA_ARGS__)
#define Error(format, ...) LogSimple(LOG_LEVEL_ERROR, COLOR_RED,    format, ## __VA_ARGS__)

/* ========== 断言 ========== */
#define Assert(cond, format, ...) \
    do { \
        if (!(cond)) { \
            printf(COLOR_RED "[ASSERT] %s:%d %s(): " format COLOR_RESET "\n", \
                   __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
            fflush(stdout); \
            extern void assert_fail_msg(void); \
            assert_fail_msg(); \
            assert(cond); \
        } \
    } while(0)

/* ========== Panic ========== */
#define Panic(format, ...) \
    do { \
        printf(COLOR_RED STYLE_BOLD "[PANIC] %s:%d %s(): " format COLOR_RESET "\n", \
               __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
        fflush(stdout); \
        extern void panic_exit(void); \
        panic_exit(); \
        exit(1); \
    } while(0)

/* ========== TODO 标记 ========== */
#define TODO() \
    do { \
        printf(COLOR_MAGENTA "[TODO] %s:%d %s()" COLOR_RESET "\n", \
               __FILE__, __LINE__, __func__); \
    } while(0)

#define TODO_MSG(format, ...) \
    do { \
        printf(COLOR_MAGENTA "[TODO] %s:%d %s(): " format COLOR_RESET "\n", \
               __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while(0)

/* ========== 条件调试 ========== */
#ifdef ENABLE_TRACE
#define Trace(...) LogTrace(__VA_ARGS__)
#else
#define Trace(...) ((void)0)
#endif

/* ========== 十六进制转储 ========== */
#define HexDump(ptr, size, format, ...) \
    do { \
        if (DEBUG_LEVEL >= LOG_LEVEL_TRACE) { \
            printf(COLOR_CYAN "[DUMP] " format COLOR_RESET "\n", ## __VA_ARGS__); \
            const uint8_t *__p = (const uint8_t *)(ptr); \
            for (size_t __i = 0; __i < (size); __i++) { \
                printf("%02x ", __p[__i]); \
                if ((__i + 1) % 16 == 0) printf("\n"); \
            } \
            if ((size) % 16 != 0) printf("\n"); \
        } \
    } while(0)

/* ========== 性能计数器 ========== */
#define TimeIt(name, code) \
    do { \
        if (DEBUG_LEVEL >= LOG_LEVEL_DEBUG) { \
            clock_t __start = clock(); \
            code \
            clock_t __end = clock(); \
            LogDebug("%s took %.3f ms", name, \
                     (double)(__end - __start) * 1000 / CLOCKS_PER_SEC); \
        } else { \
            code \
        } \
    } while(0)

/* ========== 函数入口/出口跟踪 ========== */
#ifdef ENABLE_FUNC_TRACE
#define FUNC_ENTER() LogTrace("--> enter")
#define FUNC_EXIT()  LogTrace("<-- exit")
#define FUNC_EXIT_RET(ret) LogTrace("<-- exit (ret=%d)", ret)
#else
#define FUNC_ENTER()     ((void)0)
#define FUNC_EXIT()      ((void)0)
#define FUNC_EXIT_RET(r) ((void)0)
#endif

#endif /* __DEBUG_H__ */