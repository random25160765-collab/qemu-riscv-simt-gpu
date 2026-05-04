#ifndef CONFLUX_CMD_BUILDER_H
#define CONFLUX_CMD_BUILDER_H

#include "conflux_error.h"
#include "conflux_allocator.h"
#include "conflux_event.h"
#include "conflux_queue.h"
#include "conflux_kernel.h"

/* ---- 内存拷贝请求 ---- */
typedef struct {
    uint64_t src_addr;       /* 源设备地址 */
    uint64_t dst_addr;       /* 目的设备地址 */
    size_t   size;           /* 拷贝大小 */
} conflux_copy_request_t;

/* ---- 内核执行请求 ---- */
typedef struct {
    conflux_kernel_t *kernel;   /* 内核对象 */
    
    /* 全局/局部工作大小（这里明确写出来，不直接从 kernel 读） */
    uint32_t work_dim;
    size_t   global_size[3];
    size_t   local_size[3];
    
} conflux_launch_request_t;

/* ---- 命令构建器 ---- */
typedef struct {
    conflux_queue_t    *queue;     /* 关联的命令队列 */
    conflux_allocator_t *allocator; /* 关联的显存分配器 */
    
    /* 统计 */
    uint32_t copy_cmds_built;
    uint32_t kernel_cmds_built;
    uint32_t alloc_cmds_built;
    uint32_t free_cmds_built;
} conflux_cmd_builder_t;

/* ---- API ---- */

/* 创建/销毁 */
conflux_cmd_builder_t *conflux_cmd_builder_create(conflux_queue_t *queue,
                                                 conflux_allocator_t *allocator);
void conflux_cmd_builder_destroy(conflux_cmd_builder_t *builder);

/* 构建并提交各种命令 */

/* 提交内存拷贝 */
int conflux_cmd_builder_copy(conflux_cmd_builder_t *builder,
                            const conflux_copy_request_t *req,
                            conflux_event_t **event_out);

/* 提交内核执行 */
int conflux_cmd_builder_launch(conflux_cmd_builder_t *builder,
                              const conflux_launch_request_t *req,
                              conflux_event_t **event_out);

/* 提交显存分配 */
int conflux_cmd_builder_alloc(conflux_cmd_builder_t *builder,
                             size_t size,
                             uint64_t *dev_addr_out,
                             conflux_event_t **event_out);

/* 提交显存释放 */
int conflux_cmd_builder_free(conflux_cmd_builder_t *builder,
                            uint64_t dev_addr,
                            size_t size,
                            conflux_event_t **event_out);

/* 同步屏障 */
int conflux_cmd_builder_barrier(conflux_cmd_builder_t *builder,
                               conflux_event_t **event_out);

/* 上传内核指令到设备并返回设备地址 */
int conflux_cmd_builder_upload_kernel(conflux_cmd_builder_t *builder,
                                     conflux_kernel_t *kernel,
                                     uint64_t *dev_addr_out);

/* 调试 */
void conflux_cmd_builder_dump(const conflux_cmd_builder_t *builder);

#endif