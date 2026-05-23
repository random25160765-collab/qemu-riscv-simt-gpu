/*
 * 二进制指令 Trace — 写入 fast ring buffer
 */
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include "proto.h"
#include "ring/ring.h"

static ring_buf *inst_ring;
ring_buf *event_ring;

void gpgpu_inst_trace_set_ring(ring_buf *ring)
{
    inst_ring = ring;
}

void gpgpu_event_set_ring(ring_buf *ring)
{
    event_ring = ring;
}

void gpgpu_inst_trace_bin(uint32_t inst_code, ...)
{
    if (!inst_ring) return;

    uint32_t nargs = (inst_code >> 24) & 0xF;
    uint32_t buf[16]; /* header + max 15 operands */
    buf[0] = inst_code;

    va_list args;
    va_start(args, inst_code);
    for (uint32_t i = 0; i < nargs; i++) {
        buf[i + 1] = va_arg(args, uint32_t);
    }
    va_end(args);

    ring_buf_write(inst_ring, (const uint8_t *)buf, (1 + nargs) * 4);
}

/* Write a control event (level=0x01) to the given ring buffer */
void vpu_event_write(ring_buf *ring, uint32_t event_code, ...)
{
    if (!ring) return;

    uint32_t nargs = (event_code >> 24) & 0xF;
    uint32_t buf[16];
    buf[0] = event_code;

    va_list args;
    va_start(args, event_code);
    for (uint32_t i = 0; i < nargs; i++) {
        buf[i + 1] = va_arg(args, uint32_t);
    }
    va_end(args);

    ring_buf_write(ring, (const uint8_t *)buf, (1 + nargs) * 4);
}
