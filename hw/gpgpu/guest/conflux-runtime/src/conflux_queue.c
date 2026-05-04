#include "conflux_queue.h"
#include "conflux_event.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- 内部辅助：环形缓冲区索引计算 ---- */
static inline uint32_t next_index(uint32_t idx, uint32_t size) 
{
    return (idx + 1) & (size - 1);  /* size 必须是 2 的幂 */
}

/* ---- 创建队列 ---- */
conflux_queue_t *conflux_queue_create(uint32_t ring_size,
                                     int (*execute_cmd)(conflux_cmd_t *, void *),
                                     void *device_data) 
{
    if (ring_size == 0 || (ring_size & (ring_size - 1)) != 0) {
        CONFLUX_ERROR("[QUEUE] ERROR: ring_size must be power of 2\n");
        return NULL;
    }
    
    conflux_queue_t *queue = malloc(sizeof(conflux_queue_t));
    if (!queue) return NULL;
    memset(queue, 0, sizeof(conflux_queue_t));
    
    queue->ring = malloc(sizeof(conflux_queue_entry_t) * ring_size);
    if (!queue->ring) {
        free(queue);
        return NULL;
    }
    
    queue->events = calloc(ring_size, sizeof(conflux_event_t *));
    if (!queue->events) {
        free(queue->ring);
        free(queue);
        return NULL;
    }
    
    for (uint32_t i = 0; i < ring_size; i++) {
        queue->ring[i].ready = 0;
        memset(&queue->ring[i].cmd, 0, sizeof(conflux_cmd_t));
        // 新增字段初始化
        queue->ring[i].event_owned_by_consumer = false;
    }
    
    queue->ring_size  = ring_size;
    queue->head       = 0;
    queue->tail       = 0;
    queue->execute_cmd = execute_cmd;
    queue->device_data = device_data;
    
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->not_full, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    
    CONFLUX_INFO("[QUEUE] Created, ring_size=%u (%lu bytes)\n",
                ring_size,
                (unsigned long)(sizeof(conflux_queue_entry_t) * ring_size));
    
    return queue;
}

/* ---- 销毁队列 ---- */
void conflux_queue_destroy(conflux_queue_t *queue) 
{
    if (!queue) return;
    
    conflux_queue_stop_consumer(queue);
    
    // 此时消费者已停止，可安全无锁读取（也可加锁，这里已无竞争）
    CONFLUX_INFO("[QUEUE] Destroying, submitted=%u, completed=%u\n",
                queue->submitted_count, queue->completed_count);
    
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->not_full);
    pthread_cond_destroy(&queue->not_empty);
    
    free(queue->ring);
    free(queue->events);
    free(queue);
}

/* ---- 生产者：提交命令（非阻塞） ---- */
int conflux_queue_submit(conflux_queue_t *queue,
                        conflux_cmd_t *cmd,
                        conflux_event_t **event_out) 
{
    if (!queue || !cmd) return CONFLUX_ERR_INVALID;
    
    conflux_event_t *event = conflux_event_create();
    if (!event) return CONFLUX_ERR_NOMEM;
    
    uint32_t submitted_snapshot;  // 用于锁外安全打印
    pthread_mutex_lock(&queue->lock);
    
    uint32_t next_tail = next_index(queue->tail, queue->ring_size);
    while (next_tail == queue->head) {
        CONFLUX_DEBUG("[QUEUE] Full, waiting...\n");
        pthread_cond_wait(&queue->not_full, &queue->lock);
    }
    
    conflux_queue_entry_t *entry = &queue->ring[queue->tail];
    memcpy(&entry->cmd, cmd, sizeof(conflux_cmd_t));
    
    __sync_synchronize();
    entry->ready = 1;
    
    event->command_id = queue->submitted_count + 1;
    conflux_event_set_submitted(event);
    queue->events[queue->tail] = event;
    
    // ★ 设置事件所有权标记
    if (event_out) {
        // 调用者要了事件指针，由调用者负责销毁
        entry->event_owned_by_consumer = false;
    } else {
        // 调用者不关心事件，由消费者线程负责销毁
        entry->event_owned_by_consumer = true;
    }
    
    queue->tail = next_tail;
    queue->submitted_count++;
    submitted_snapshot = queue->submitted_count;   // 保存快照
    queue->last_error = CONFLUX_SUCCESS;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->lock);
    
    if (event_out) {
        *event_out = event;
    }
    // ★ 不再有 else 销毁，事件由消费者按标记销毁
    
    CONFLUX_DEBUG("[QUEUE] Submitted cmd type=%d, total=%u\n",
                 cmd->type, submitted_snapshot);   // 使用锁内快照
    
    return CONFLUX_SUCCESS;
}

/* ---- 生产者：提交并阻塞等待完成 ---- */
int conflux_queue_submit_sync(conflux_queue_t *queue, conflux_cmd_t *cmd) 
{
    conflux_event_t *event = NULL;
    
    int ret = conflux_queue_submit(queue, cmd, &event);
    if (ret != CONFLUX_SUCCESS) return ret;
    
    conflux_event_wait(event, 0);
    conflux_event_destroy(event);   // 同步等待结束后调用者销毁
    
    return CONFLUX_SUCCESS;
}

/* ---- 生产者：排空队列 ---- */
int conflux_queue_drain(conflux_queue_t *queue) 
{
    if (!queue) return CONFLUX_ERR_INVALID;
    
    pthread_mutex_lock(&queue->lock);
    while (queue->head != queue->tail) {
        // ★ 直接计算 pending，避免调用已加锁的 pending_count 函数
        uint32_t pending = (queue->tail - queue->head) & (queue->ring_size - 1);
        CONFLUX_DEBUG("[QUEUE] Draining: pending=%u\n", pending);
        pthread_cond_wait(&queue->not_full, &queue->lock);
    }
    pthread_mutex_unlock(&queue->lock);
    
    return CONFLUX_SUCCESS;
}

/* ---- 消费者线程 ---- */
static void *consumer_loop(void *arg) 
{
    conflux_queue_t *queue = (conflux_queue_t *)arg;
    
    CONFLUX_DEBUG("[CONSUMER] Thread started\n");
    
    while (1) {
        pthread_mutex_lock(&queue->lock);
        
        /* 等待队列不为空 */
        while (queue->head == queue->tail) {
            /* 队列空且消费者需要停止 -> 退出 */
            if (!queue->consumer_running) {
                pthread_mutex_unlock(&queue->lock);
                CONFLUX_DEBUG("[CONSUMER] Thread stopped, completed=%u\n", queue->completed_count);
                return NULL;
            }
            pthread_cond_wait(&queue->not_empty, &queue->lock);
        }
        
        /* 此时队列必非空，拿出一个命令（无论 consumer_running 状态） */
        uint32_t head = queue->head;
        conflux_queue_entry_t *entry = &queue->ring[head];
        conflux_cmd_t cmd = entry->cmd;
        conflux_event_t *completed_event = queue->events[head];
        bool should_destroy = entry->event_owned_by_consumer;
        
        entry->ready = 0;
        memset(&entry->cmd, 0, sizeof(conflux_cmd_t));
        entry->event_owned_by_consumer = false;
        queue->events[head] = NULL;
        
        queue->head = next_index(head, queue->ring_size);
        pthread_cond_signal(&queue->not_full);
        pthread_mutex_unlock(&queue->lock);
        
        /* 执行命令（可能耗时） */
        int exec_ret = 0;
        if (cmd.type != CONFLUX_CMD_NOP && queue->execute_cmd) {
            exec_ret = queue->execute_cmd(&cmd, queue->device_data);
        }
        
        /* 先更新统计：必须在 event_set_complete 之前发生，
         * 否则等待 event 的线程读 completed_count 会是旧值 */
        pthread_mutex_lock(&queue->lock);
        queue->completed_count++;
        if (queue->last_error == CONFLUX_SUCCESS && exec_ret != 0) {
            queue->last_error = exec_ret;
        }
        pthread_mutex_unlock(&queue->lock);

        /* 通知事件完成 / 销毁（happens-after 计数器更新） */
        if (completed_event) {
            if (exec_ret == 0) {
                conflux_event_set_complete(completed_event);
            } else {
                conflux_event_set_failed(completed_event, exec_ret);
            }
            if (should_destroy) {
                conflux_event_destroy(completed_event);
            }
        }
    }
}

/* ---- 启动消费者 ---- */
int conflux_queue_start_consumer(conflux_queue_t *queue) 
{
    if (!queue) return CONFLUX_ERR_INVALID;
    
    if (queue->consumer_running) {
        return CONFLUX_SUCCESS;
    }
    
    queue->consumer_running = 1;
    
    int ret = pthread_create(&queue->consumer_thread, 
                             NULL, consumer_loop, queue);
    if (ret != 0) {
        queue->consumer_running = 0;
        CONFLUX_ERROR("[QUEUE] Failed to create consumer thread\n");
        return CONFLUX_ERR_NOMEM;
    }
    
    return CONFLUX_SUCCESS;
}

/* ---- 停止消费者 ---- */
int conflux_queue_stop_consumer(conflux_queue_t *queue) 
{
    if (!queue) return CONFLUX_ERR_INVALID;
    
    if (!queue->consumer_running) {
        return CONFLUX_SUCCESS;
    }
    
    pthread_mutex_lock(&queue->lock);
    queue->consumer_running = 0;
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->lock);
    
    pthread_join(queue->consumer_thread, NULL);
    
    return CONFLUX_SUCCESS;
}

/* ---- 查询（线程安全版，去掉 const） ---- */
int conflux_queue_is_empty(conflux_queue_t *queue) 
{
    if (!queue) return 1;
    pthread_mutex_lock(&queue->lock);
    int empty = (queue->head == queue->tail);
    pthread_mutex_unlock(&queue->lock);
    return empty;
}

int conflux_queue_is_full(conflux_queue_t *queue) 
{
    if (!queue) return 1;
    pthread_mutex_lock(&queue->lock);
    int full = (next_index(queue->tail, queue->ring_size) == queue->head);
    pthread_mutex_unlock(&queue->lock);
    return full;
}

uint32_t conflux_queue_pending_count(conflux_queue_t *queue) 
{
    if (!queue) return 0;
    pthread_mutex_lock(&queue->lock);
    uint32_t pending = (queue->tail - queue->head) & (queue->ring_size - 1);
    pthread_mutex_unlock(&queue->lock);
    return pending;
}

uint32_t conflux_queue_get_completed(conflux_queue_t *queue) {
    if (!queue) return 0;
    pthread_mutex_lock(&queue->lock);
    uint32_t c = queue->completed_count;
    pthread_mutex_unlock(&queue->lock);
    return c;
}

uint32_t conflux_queue_get_submitted(conflux_queue_t *queue) {
    if (!queue) return 0;
    pthread_mutex_lock(&queue->lock);
    uint32_t s = queue->submitted_count;
    pthread_mutex_unlock(&queue->lock);
    return s;
}

/* ---- 调试 dump，printf 完全保留 ---- */
void conflux_queue_dump(conflux_queue_t *queue) 
{
    if (!queue) {
        printf("NULL queue\n");
        return;
    }
    
    pthread_mutex_lock(&queue->lock);
    
    printf("\n=== Queue State ===\n");
    printf("  ring_size:  %u\n", queue->ring_size);
    printf("  head:       %u\n", queue->head);
    printf("  tail:       %u\n", queue->tail);
    // 直接计算，避免调用加锁版本的 pending_count
    printf("  pending:    %u\n", (queue->tail - queue->head) & (queue->ring_size - 1));
    printf("  submitted:  %u\n", queue->submitted_count);
    printf("  completed:  %u\n", queue->completed_count);
    printf("  consumer:   %s\n", queue->consumer_running ? "running" : "stopped");
    printf("  last_error: %d\n", queue->last_error);
    
    printf("\n  Ring buffer:\n");
    for (uint32_t i = 0; i < queue->ring_size; i++) {
        printf("    [%2u] ready=%u type=%u\n",
               i, queue->ring[i].ready, queue->ring[i].cmd.type);
    }
    
    pthread_mutex_unlock(&queue->lock);
}