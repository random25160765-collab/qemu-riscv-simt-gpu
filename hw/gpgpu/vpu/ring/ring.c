#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdatomic.h>
#include <sys/uio.h>
#include <stdlib.h>

#include "ring.h"

ring_buf *ring_buf_create(size_t size)
{
    ring_buf *rb = malloc(sizeof(ring_buf));
    if (!rb) return NULL;

    uint8_t *mem = malloc(size);
    if (!mem) {
        free(rb);
        return NULL;
    }

    rb->buf = mem;
    rb->size = size;
    atomic_init(&rb->r, 0);
    atomic_init(&rb->w, 0);
    return rb;
}

void ring_buf_init(ring_buf *rb, uint8_t *mem, size_t size)
{
    rb->buf = mem;
    rb->size = size;
    atomic_init(&rb->r, 0);
    atomic_init(&rb->w, 0);
}

void ring_buf_destroy(ring_buf *rb)
{
    if (rb) {
        free(rb->buf);
        free(rb);
    }
}

int ring_buf_peek(ring_buf *rb, struct iovec iov[2])
{
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    size_t available = w - r;
    if (available == 0) return 0;
    if (available > rb->size) available = rb->size;

    uint32_t r_phy = r & (rb->size - 1);
    size_t first_len = rb->size - r_phy;

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

void ring_buf_commit(ring_buf *rb, size_t len)
{
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);
    atomic_store_explicit(&rb->r, r + len, memory_order_release);
}

int ring_buf_write(ring_buf *rb, const uint8_t *src, size_t len)
{
    if (len == 0) return 0;
    if (len > rb->size) {
        src += len - rb->size;
        len = rb->size;
    }

    uint32_t r = atomic_load_explicit(&rb->r, memory_order_acquire);
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_relaxed);

    size_t available = rb->size - (w - r);
    if (available < len) {
        size_t new_r = r + len - available;
        atomic_store_explicit(&rb->r, new_r, memory_order_release);
        r = new_r;
    }

    uint32_t w_phy = w & (rb->size - 1);
    size_t first_write = rb->size - w_phy;
    if (len <= first_write) {
        memcpy(&rb->buf[w_phy], src, len);
    } else {
        memcpy(&rb->buf[w_phy], src, first_write);
        memcpy(&rb->buf[0], src + first_write, len - first_write);
    }
    atomic_store_explicit(&rb->w, w + len, memory_order_release);

    return 0;
}

size_t ring_buf_read(ring_buf *rb, uint8_t *dst, size_t len)
{
    if (len == 0) return 0;
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    size_t available = w - r;
    if (available < len) len = available;

    uint32_t r_phy = r & (rb->size - 1);
    size_t first_read = rb->size - r_phy;
    if (len <= first_read) {
        memcpy(dst, &rb->buf[r_phy], len);
    } else {
        memcpy(dst, &rb->buf[r_phy], first_read);
        memcpy(dst + first_read, &rb->buf[0], len - first_read);
    }
    atomic_store_explicit(&rb->r, r + len, memory_order_release);

    return len;
}
