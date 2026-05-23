#ifndef VPU_RING_H
#define VPU_RING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <sys/uio.h>

typedef struct {
    uint8_t *buf;
    size_t size;
    _Atomic uint32_t r;
    _Atomic uint32_t w;
} ring_buf;

/* Allocate and initialize a ring buffer of given size (must be power of 2) */
ring_buf *ring_buf_create(size_t size);

/* Initialize a ring buffer with external backing memory */
void ring_buf_init(ring_buf *rb, uint8_t *mem, size_t size);

/* Destroy a ring buffer allocated by ring_buf_create */
void ring_buf_destroy(ring_buf *rb);

/* Peek available data without consuming; returns number of iovec entries (0, 1, or 2) */
int ring_buf_peek(ring_buf *rb, struct iovec iov[2]);

/* Commit consumed data after peek */
void ring_buf_commit(ring_buf *rb, size_t len);

/* Write data to ring; overwrites oldest data if full */
int ring_buf_write(ring_buf *rb, const uint8_t *src, size_t len);

/* Read data from ring; returns actual bytes read */
size_t ring_buf_read(ring_buf *rb, uint8_t *dst, size_t len);

#endif /* VPU_RING_H */
