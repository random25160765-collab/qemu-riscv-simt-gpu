#ifndef CONFLUX_EVENT_H
#define CONFLUX_EVENT_H

#include "conflux_error.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

/* OpenCL的异步命令 */
/* 提交后不用等待完成，用一个事件对象来查询/等待 */

/* 事件状态 */
typedef enum {
    CONFLUX_EVENT_QUEUED     = 0,  /* 刚创建，在队列里 */
    CONFLUX_EVENT_SUBMITTED  = 1,  /* 已提交给设备 */
    CONFLUX_EVENT_RUNNING    = 2,  /* 设备正在执行 */
    CONFLUX_EVENT_COMPLETE   = 3,  /* 执行完成 */
    CONFLUX_EVENT_FAILED     = 4,  /* 执行失败 */
} conflux_event_status_t;

/* 事件对象 */
typedef struct _conflux_event {
    /* 状态 */
    conflux_event_status_t status;
    int error_code;              /* 失败时的错误码 */
    
    /* 时间戳（纳秒） */
    // ? 溢出问题
    uint64_t time_queued;        /* 创建时刻 */
    uint64_t time_submitted;     /* 提交时刻 */
    uint64_t time_started;       /* 开始执行时刻 */
    uint64_t time_ended;         /* 完成时刻 */
    
    /* 关联信息 */
    uint64_t command_id;         /* 关联的命令ID */
    void *user_data;             /* 用户自定义数据 */
    
    /* 同步原语 */
    pthread_mutex_t lock;        /* 保护状态字段 */
    pthread_cond_t  cond;        /* 用于等待完成 */
    
    /* 回调（完成时调用） */
    void (*callback)(struct _conflux_event *, void *);
    void *callback_data;
    
    /* 引用计数 */
    int ref_count;
} conflux_event_t;

/* ---------- API ---------- */

/* 创建/销毁 */
conflux_event_t *conflux_event_create(void);
void conflux_event_destroy(conflux_event_t *event);

/* 状态转换（事件的生命周期） */
void conflux_event_set_queued(conflux_event_t *event);
void conflux_event_set_submitted(conflux_event_t *event);
void conflux_event_set_running(conflux_event_t *event);
void conflux_event_set_complete(conflux_event_t *event);
void conflux_event_set_failed(conflux_event_t *event, int error_code);

/* 查询状态 */
conflux_event_status_t conflux_event_get_status(conflux_event_t *event);
int conflux_event_is_complete(conflux_event_t *event);

/* 等待（阻塞直到完成） */
conflux_error_t conflux_event_wait(conflux_event_t *event, uint64_t timeout_ns);

/* 增加/减少引用计数 */
void conflux_event_retain(conflux_event_t *event);
void conflux_event_release(conflux_event_t *event);

/* 设置回调 */
void conflux_event_set_callback(conflux_event_t *event,
                               void (*callback)(conflux_event_t *, void *),
                               void *data);

/* 调试 */
void conflux_event_dump(const conflux_event_t *event, char *buf, size_t buf_size);

/* 获取当前时间（纳秒）的辅助函数 */
uint64_t conflux_get_time_ns(void);

#endif