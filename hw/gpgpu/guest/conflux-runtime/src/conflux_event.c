#include "conflux_event.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset, snprintf */
#include <stdio.h>    /* snprintf */
#include <time.h>     /* clock_gettime */
#include <unistd.h>
#include <errno.h>

/* 获取当前纳秒时间 */
uint64_t conflux_get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* 创建/销毁 */
conflux_event_t *conflux_event_create(void)
{
    conflux_event_t *event = malloc(sizeof(conflux_event_t));
    if (!event) {
        return NULL;
    }

    memset(event, 0, sizeof(conflux_event_t));

    pthread_mutex_init(&event->lock, NULL);
    pthread_cond_init(&event->cond, NULL);

    event->status = CONFLUX_EVENT_QUEUED;
    event->time_queued = conflux_get_time_ns();
    event->ref_count = 1;

    CONFLUX_DEBUG("[EVENT] Created, id=%p\n", (void *)event);

    return event;
}

void conflux_event_destroy(conflux_event_t *event)
{
    if (!event) return;

    CONFLUX_DEBUG("[EVENT] Destroying, id=%p\n", (void *)event);

    pthread_mutex_destroy(&event->lock);
    pthread_cond_destroy(&event->cond);

    free(event);
}

/* 状态转换（事件的生命周期） */
void conflux_event_set_queued(conflux_event_t *event)
{
    if (!event) return;

    pthread_mutex_lock(&event->lock);

    event->status = CONFLUX_EVENT_QUEUED;
    event->time_queued = conflux_get_time_ns();

    pthread_mutex_unlock(&event->lock);
}

void conflux_event_set_submitted(conflux_event_t *event)
{
    if (!event) return;

    pthread_mutex_lock(&event->lock);

    event->status = CONFLUX_EVENT_SUBMITTED;
    event->time_submitted = conflux_get_time_ns();

    pthread_mutex_unlock(&event->lock);

    CONFLUX_DEBUG("[EVENT] %p -> SUBMITTED\n", (void *)event);
}

void conflux_event_set_running(conflux_event_t *event)
{
    if (!event) return;

    pthread_mutex_lock(&event->lock);

    event->status = CONFLUX_EVENT_RUNNING;
    event->time_started = conflux_get_time_ns();

    pthread_mutex_unlock(&event->lock);

    CONFLUX_DEBUG("[EVENT] %p -> RUNNING\n", (void *)event);
}

void conflux_event_set_complete(conflux_event_t *event)
{
    if (!event) return;

    pthread_mutex_lock(&event->lock);

    event->status = CONFLUX_EVENT_COMPLETE;
    event->time_ended = conflux_get_time_ns();

    pthread_cond_broadcast(&event->cond);
    
    void (*cb)(conflux_event_t *, void *) = event->callback;
    void *cb_data = event->callback_data;

    pthread_mutex_unlock(&event->lock);

    CONFLUX_DEBUG("[EVENT] %p -> COMPLETE\n", (void *)event);

    if (cb) {
        cb(event, cb_data);
    }
}

void conflux_event_set_failed(conflux_event_t *event, int error_code)
{
    if (!event) return;

    pthread_mutex_lock(&event->lock);

    event->status = CONFLUX_EVENT_FAILED;
    event->error_code = error_code;
    event->time_ended = conflux_get_time_ns();

    pthread_cond_broadcast(&event->cond);
    
    void (*cb)(conflux_event_t *, void *) = event->callback;
    void *cb_data = event->callback_data;

    pthread_mutex_unlock(&event->lock);

    CONFLUX_ERROR("[EVENT] %p -> FAILED (error=%d)\n", (void *)event, error_code);

    if (cb) {
        cb(event, cb_data);
    }
}

/* 查询状态 */
conflux_event_status_t conflux_event_get_status(conflux_event_t *event)
{
    if (!event) return CONFLUX_EVENT_FAILED;
    
    pthread_mutex_lock(&event->lock);
    conflux_event_status_t s = event->status;
    pthread_mutex_unlock(&event->lock);
    
    return s;
}

int conflux_event_is_complete(conflux_event_t *event)
{
    if (!event) return -1;
    
    pthread_mutex_lock(&event->lock);
    int done = (event->status == CONFLUX_EVENT_COMPLETE ||
                event->status == CONFLUX_EVENT_FAILED);
    pthread_mutex_unlock(&event->lock);
    
    return done;
}

/* 等待（阻塞直到完成） */
conflux_error_t conflux_event_wait(conflux_event_t *event, uint64_t timeout_ns)
{
    if (!event) return -1;

    pthread_mutex_lock(&event->lock);

    /* 如果已经完成，直接返回 */
    if (event->status == CONFLUX_EVENT_COMPLETE ||
        event->status == CONFLUX_EVENT_FAILED) {
        pthread_mutex_unlock(&event->lock);
        return CONFLUX_SUCCESS;
    }  

    if (timeout_ns == 0) {
        /* 无超时 = 无限等待 */
        CONFLUX_DEBUG("[EVENT] %p: waiting indefinitely...\n", (void *)event);
        pthread_cond_wait(&event->cond, &event->lock);
        pthread_mutex_unlock(&event->lock);
        return CONFLUX_SUCCESS;
    } else {
        /* 带超时的等待 */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        
        uint64_t total_ns = (uint64_t)ts.tv_sec * 1000000000ULL 
                            + ts.tv_nsec + timeout_ns;
        ts.tv_sec  = total_ns / 1000000000ULL;
        ts.tv_nsec = total_ns % 1000000000ULL;
        
        CONFLUX_DEBUG("[EVENT] %p: waiting with timeout=%lu ns...\n", 
                     (void *)event, (unsigned long)timeout_ns);
        
        int ret = pthread_cond_timedwait(&event->cond, &event->lock, &ts);
        pthread_mutex_unlock(&event->lock);
        
        if (ret == ETIMEDOUT) {
            CONFLUX_WARN("[EVENT] %p: wait TIMED OUT\n", (void *)event);
            return CONFLUX_ERR_TIMEOUT;  /* 超时 */
        }
        return CONFLUX_SUCCESS;  /* 被唤醒 */
    }
}

/* 增加/减少引用计数 */
void conflux_event_retain(conflux_event_t *event)
{
    if (!event) return;
    
    pthread_mutex_lock(&event->lock);
    event->ref_count++;
    CONFLUX_DEBUG("[EVENT] %p: retain → ref_count=%d\n", 
                 (void *)event, event->ref_count);
    pthread_mutex_unlock(&event->lock);
}

void conflux_event_release(conflux_event_t *event)
{
    if (!event) return;
    
    pthread_mutex_lock(&event->lock);
    event->ref_count--;
    CONFLUX_DEBUG("[EVENT] %p: release → ref_count=%d\n", 
                 (void *)event, event->ref_count);
    
    int should_free = (event->ref_count <= 0);
    pthread_mutex_unlock(&event->lock);
    
    if (should_free) {
        conflux_event_destroy(event);
    }
}

/* 设置回调 */
void conflux_event_set_callback(conflux_event_t *event,
                               void (*callback)(conflux_event_t *, void *),
                               void *data)
{
    if (!event) return;
    
    pthread_mutex_lock(&event->lock);
    event->callback      = callback;
    event->callback_data = data;
    pthread_mutex_unlock(&event->lock);
    
    CONFLUX_DEBUG("[EVENT] %p: callback set\n", (void *)event);
}

/* 调试 dump 函数 —— 无 printf，无需修改 */
void conflux_event_dump(const conflux_event_t *event, char *buf, size_t buf_size)
{
    if (!event) {
        snprintf(buf, buf_size, "NULL event");
        return;
    }
    
    /* 不加锁，调试用 */
    const char *status_str = "UNKNOWN";
    switch (event->status) {
        case CONFLUX_EVENT_QUEUED:    status_str = "QUEUED";    break;
        case CONFLUX_EVENT_SUBMITTED: status_str = "SUBMITTED"; break;
        case CONFLUX_EVENT_RUNNING:   status_str = "RUNNING";   break;
        case CONFLUX_EVENT_COMPLETE:  status_str = "COMPLETE";  break;
        case CONFLUX_EVENT_FAILED:    status_str = "FAILED";    break;
    }
    
    snprintf(buf, buf_size,
             "Event %p:\n"
             "  status:    %s\n"
             "  error:     %d\n"
             "  cmd_id:    %lu\n"
             "  ref_count: %d\n"
             "  timeline (ns):\n"
             "    queued:    %lu\n"
             "    submitted: %lu\n"
             "    started:   %lu\n"
             "    ended:     %lu\n"
             "  duration:   %lu ns",
             (void *)event,
             status_str,
             event->error_code,
             (unsigned long)event->command_id,
             event->ref_count,
             (unsigned long)event->time_queued,
             (unsigned long)event->time_submitted,
             (unsigned long)event->time_started,
             (unsigned long)event->time_ended,
             (unsigned long)(event->time_ended - event->time_started));
}