#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <sys/uio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#define SIZE (4 * 1024) // 加括号

#if (SIZE & (SIZE - 1)) != 0
#error "SIZE must be a power of 2"
#endif

typedef struct {
    uint8_t buf[SIZE];
    _Atomic uint32_t r;
    _Atomic uint32_t w;
} ring_buf;

void init_ring_buf(ring_buf* rb) {
    atomic_init(&rb->r, 0);
    atomic_init(&rb->w, 0);
}

int log_read_peek(ring_buf* rb, struct iovec iov[2]) {

    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    size_t available = w - r;
    if (available == 0) return 0;
    if (available > SIZE) available = SIZE; // 并发防御

    uint32_t r_phy = r & (SIZE - 1);
    size_t first_len = SIZE - r_phy;

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

void log_read_commit(ring_buf* rb, size_t len) {
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);
    atomic_store_explicit(&rb->r, r + len, memory_order_release);
}

int log_write(const uint8_t* src, size_t len, ring_buf* rb) {

    if (len == 0) return 0;
    if (len > SIZE) {
        src += len - SIZE;
        len = SIZE;
    }

    uint32_t r = atomic_load_explicit(&rb->r, memory_order_acquire);
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_relaxed);

    /* 判断有多少空间可写 */
    size_t available = SIZE - (w - r);
    if (available < len) {
        size_t new_r = r + len - available;
        atomic_store_explicit(&rb->r, new_r, memory_order_release);
        r = new_r;
    }

    /* 分两次传入数据 */
    uint32_t w_phy = w & (SIZE - 1);
    size_t first_write = SIZE - w_phy;
    if (len <= first_write) {
        memcpy(&rb->buf[w_phy], src, len);
    } else {
        memcpy(&rb->buf[w_phy], src, first_write);
        memcpy(&rb->buf[0], src + first_write, len - first_write);
    }
    atomic_store_explicit(&rb->w, w + len, memory_order_release);

    return 0;
}

size_t log_read(uint8_t* dst, size_t len, ring_buf* rb) {

    if (len == 0) return 0;
    uint32_t w = atomic_load_explicit(&rb->w, memory_order_acquire);
    uint32_t r = atomic_load_explicit(&rb->r, memory_order_relaxed);

    /* 判断有多少空间可读 */
    size_t available = w - r;
    if (available < len) len = available;

    /* 分两次读 */
    uint32_t r_phy = r & (SIZE - 1);
    size_t first_read = SIZE - r_phy;
    if (len <= first_read) {
        memcpy(dst, &rb->buf[r_phy], len);
    } else {
        memcpy(dst, &rb->buf[r_phy], first_read);
        memcpy(dst + first_read, &rb->buf[0], len - first_read);
    }
    atomic_store_explicit(&rb->r, r + len, memory_order_release);

    return len;
}