#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <errno.h>

#include "probe.h"
#include "ring/ring.h"

static struct {
    int slow_fd;
    int fast_fd;
    int slow_client;
    int fast_client;
    ring_buf *slow_ring;
    ring_buf *fast_ring;
    const char *slow_path;
    const char *fast_path;
    pthread_t slow_thread;
    pthread_t fast_thread;
    volatile bool running;
} probe;

static int create_server(const char *path)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("probe: socket");
        return -1;
    }

    unlink(path);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("probe: bind");
        close(fd);
        return -1;
    }

    if (listen(fd, 1) < 0) {
        perror("probe: listen");
        close(fd);
        return -1;
    }

    return fd;
}

static int accept_client(int server_fd)
{
    int fd = accept(server_fd, NULL, NULL);
    if (fd < 0) {
        if (errno != EINTR)
            perror("probe: accept");
        return -1;
    }
    return fd;
}

static int send_frame(int fd, const uint8_t *data, size_t len)
{
    uint32_t frame_len = (uint32_t)len;
    struct iovec iov[2];
    iov[0].iov_base = &frame_len;
    iov[0].iov_len = 4;
    iov[1].iov_base = (void *)data;
    iov[1].iov_len = len;

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = 2;

    ssize_t sent = sendmsg(fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        if (errno == EPIPE || errno == ECONNRESET)
            return -2;  /* client disconnected */
        perror("probe: sendmsg");
        return -1;
    }
    return 0;
}

static int drain_and_send(ring_buf *rb, int client_fd)
{
    struct iovec iov[2];
    int n = ring_buf_peek(rb, iov);
    if (n <= 0) return 0;

    size_t total = 0;
    for (int i = 0; i < n; i++)
        total += iov[i].iov_len;

    if (total == 0) return 0;

    int rc = 0;
    if (client_fd >= 0) {
        if (n == 1) {
            rc = send_frame(client_fd, iov[0].iov_base, iov[0].iov_len);
        } else {
            uint8_t buf[65536];
            size_t copy_len = total > sizeof(buf) ? sizeof(buf) : total;
            size_t off = 0;
            for (int i = 0; i < n; i++) {
                size_t chunk = iov[i].iov_len;
                if (off + chunk > copy_len) chunk = copy_len - off;
                memcpy(buf + off, iov[i].iov_base, chunk);
                off += chunk;
            }
            rc = send_frame(client_fd, buf, copy_len);
        }
    }

    /* Only commit if send succeeded. On disconnect (rc == -2), leave
     * data in the ring so it can be sent to the next client. */
    if (rc == 0)
        ring_buf_commit(rb, total);

    return rc;
}

static void *slow_thread_fn(void *arg)
{
    (void)arg;
    while (probe.running) {
        if (probe.slow_client < 0)
            probe.slow_client = accept_client(probe.slow_fd);
        if (drain_and_send(probe.slow_ring, probe.slow_client) == -2) {
            close(probe.slow_client);
            probe.slow_client = -1;
        }
        usleep(1000);
    }
    return NULL;
}

static void *fast_thread_fn(void *arg)
{
    (void)arg;
    while (probe.running) {
        if (probe.fast_client < 0)
            probe.fast_client = accept_client(probe.fast_fd);
        if (drain_and_send(probe.fast_ring, probe.fast_client) == -2) {
            close(probe.fast_client);
            probe.fast_client = -1;
        }
        usleep(1000);
    }
    return NULL;
}

int probe_init(ProbeConfig *cfg)
{
    memset(&probe, 0, sizeof(probe));
    probe.slow_ring = cfg->slow_ring;
    probe.fast_ring = cfg->fast_ring;
    probe.slow_path = cfg->slow_path;
    probe.fast_path = cfg->fast_path;
    probe.slow_client = -1;
    probe.fast_client = -1;

    probe.slow_fd = create_server(cfg->slow_path);
    if (probe.slow_fd < 0) return -1;

    probe.fast_fd = create_server(cfg->fast_path);
    if (probe.fast_fd < 0) {
        close(probe.slow_fd);
        unlink(cfg->slow_path);
        return -1;
    }

    return 0;
}

void probe_start(void)
{
    probe.running = true;
    pthread_create(&probe.slow_thread, NULL, slow_thread_fn, NULL);
    pthread_create(&probe.fast_thread, NULL, fast_thread_fn, NULL);
}

void probe_stop(void)
{
    probe.running = false;
    pthread_join(probe.slow_thread, NULL);
    pthread_join(probe.fast_thread, NULL);

    if (probe.slow_client >= 0) close(probe.slow_client);
    if (probe.fast_client >= 0) close(probe.fast_client);
    close(probe.slow_fd);
    close(probe.fast_fd);
    unlink(probe.slow_path);
    unlink(probe.fast_path);
}
