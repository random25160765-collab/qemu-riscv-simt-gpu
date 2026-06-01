/*
 * 观测协议 — 独立版（输出到屏幕）
 *
 * 在独立性能测试中，trace/event 直接打印到 stderr，
 * 无需 ring buffer 依赖。
 */
#ifndef GPGPU_PROTO_H
#define GPGPU_PROTO_H

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include "proto/pt_inst.h"
#include "proto/pt_event.h"

/* 控制 trace 详细程度：
 *   0 = 只打印 error event
 *   1 = 打印所有 event
 *   2 = 打印所有 event + inst trace
 */
#ifndef TRACE_LEVEL
#define TRACE_LEVEL 0
#endif

/* ============================================================
 * 函数声明
 * ============================================================ */
void gpgpu_inst_trace_set_ring(void *ring);
void gpgpu_event_set_ring(void *ring);
void gpgpu_inst_trace_bin(uint32_t inst_code, ...);
void vpu_event_write(uint32_t event_code, ...);

/* ============================================================
 * 宏定义
 * ============================================================ */
#define GPGPU_INST_BIN(inst_code, ...)                                        \
    do {                                                                      \
        if (TRACE_LEVEL >= 2) gpgpu_inst_trace_bin(inst_code, ##__VA_ARGS__); \
    } while (0)

#define GPGPU_EVENT(ring, event_code, ...)                                \
    do {                                                                  \
        if (TRACE_LEVEL >= 1) vpu_event_write(event_code, ##__VA_ARGS__); \
    } while (0)

#endif /* GPGPU_PROTO_H */
