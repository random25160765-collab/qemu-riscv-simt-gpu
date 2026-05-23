/*
 * 二进制观测协议
 */
#ifndef GPGPU_PROTO_H
#define GPGPU_PROTO_H

#include <stdint.h>
#include "proto/pt_inst.h"
#include "proto/pt_event.h"

/* Forward declaration */
typedef struct ring_buf ring_buf;

/* Global ring buffer for control events (level=0x01, slow ring) */
extern ring_buf *event_ring;

/* Set the ring buffer for instruction trace (level=0x00, fast ring) */
void gpgpu_inst_trace_set_ring(ring_buf *ring);

/* Set the ring buffer for control events (level=0x01, slow ring) */
void gpgpu_event_set_ring(ring_buf *ring);

/* Write an instruction trace record */
void gpgpu_inst_trace_bin(uint32_t inst_code, ...);

/* Write a control event record to the given ring buffer */
void vpu_event_write(ring_buf *ring, uint32_t event_code, ...);

/* Convenience macros */
#define GPGPU_INST_BIN(inst_code, ...) \
    gpgpu_inst_trace_bin(inst_code, ##__VA_ARGS__)

#define VPU_EVENT(ring, event_code, ...) \
    vpu_event_write(ring, event_code, ##__VA_ARGS__)

#endif /* GPGPU_PROTO_H */
