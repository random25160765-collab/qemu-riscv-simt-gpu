#ifndef CONFLUX_QUEUE_H
#define CONFLUX_QUEUE_H

#include "conflux_error.h"
#include "conflux_event.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <stdbool.h>

/* ---- 命令类型 ---- */
typedef enum {
    CONFLUX_CMD_NOP      = 0,  /* 空操作 */
    CONFLUX_CMD_COPY     = 1,  /* 内存拷贝 */
    CONFLUX_CMD_KERNEL   = 2,  /* 执行内核 */
    CONFLUX_CMD_ALLOC    = 3,  /* 分配设备内存 */
    CONFLUX_CMD_FREE     = 4,  /* 释放设备内存 */
    CONFLUX_CMD_BARRIER  = 5,  /* 同步屏障 */
} conflux_cmd_type_t;

/* ---- 命令描述符 ---- */
typedef struct {
    conflux_cmd_type_t type;       /* 命令类型 */
    uint32_t reserved0;
    uint64_t src_addr;            /* 源地址 */
    uint64_t dst_addr;            /* 目的地址 */
    uint32_t size;                /* 大小/参数 */
    uint32_t flags;               /* 标志位 */
    uint32_t kernel_id;           /* 内核 ID（CMD_KERNEL 时用） */
    uint32_t reserved1;            /* 对齐填充 */
} conflux_cmd_t;

_Static_assert(sizeof(conflux_cmd_t) == 40, "conflux_cmd_t size must be 40");

/* ---- 队列条目 ---- */
typedef struct {
    volatile uint32_t ready;   /* 生产者写 1 表示有命令，消费者读完后写 0 */
    uint32_t padding0;
    conflux_cmd_t cmd;          /* 命令内容 */
    bool event_owned_by_consumer; 
    char padding1[15];
}  conflux_queue_entry_t;

_Static_assert(sizeof(conflux_queue_entry_t) == 64, 
               "queue entry must be 64 bytes (cache line)");

/* ---- 命令队列 ---- */
typedef struct {
    /* 环形缓冲区 */
    conflux_queue_entry_t *ring;       /* 环形缓冲区基址 */
    conflux_event_t      **events;
    uint32_t ring_size;               /* 条目数（必须是 2 的幂） */
    volatile uint32_t head;           /* 消费者索引（设备读） */
    volatile uint32_t tail;           /* 生产者索引（主机写） */
    
    /* 统计 */
    uint32_t submitted_count;         /* 已提交命令数 */
    uint32_t completed_count;         /* 已完成命令数 */
    
    /* 同步 */
    pthread_mutex_t lock;             /* 保护队列字段 */
    pthread_cond_t  not_full;         /* 队列满时生产者等待 */
    pthread_cond_t  not_empty;        /* 队列空时消费者等待 */
    
    /* 设备回调（消费者线程调用） */
    int (*execute_cmd)(conflux_cmd_t *cmd, void *user_data);
    void *device_data;
    
    /* 消费者线程 */
    pthread_t consumer_thread;
    int consumer_running;
    
    /* 错误码 */
    int last_error;
} conflux_queue_t;

/* ---- API ---- */

/* 创建/销毁 */
conflux_queue_t *conflux_queue_create(uint32_t ring_size,
                                     int (*execute_cmd)(conflux_cmd_t *, void *),
                                     void *device_data);
void conflux_queue_destroy(conflux_queue_t *queue);

/* 生产者：提交命令（非阻塞） */
int conflux_queue_submit(conflux_queue_t *queue, 
                        conflux_cmd_t *cmd,
                        conflux_event_t **event_out);

/* 生产者：提交命令并阻塞等待完成 */
int conflux_queue_submit_sync(conflux_queue_t *queue,
                             conflux_cmd_t *cmd);

/* 生产者：等待所有提交的命令完成 */
int conflux_queue_drain(conflux_queue_t *queue);

/* 消费者：启动/停止消费者线程 */
int conflux_queue_start_consumer(conflux_queue_t *queue);
int conflux_queue_stop_consumer(conflux_queue_t *queue);

/* 查询 */
int conflux_queue_is_empty(conflux_queue_t *queue);
int conflux_queue_is_full(conflux_queue_t *queue);
uint32_t conflux_queue_pending_count(conflux_queue_t *queue);
uint32_t conflux_queue_get_completed(conflux_queue_t *queue);
uint32_t conflux_queue_get_submitted(conflux_queue_t *queue);

/* 调试 */
void conflux_queue_dump(conflux_queue_t *queue);

#endif