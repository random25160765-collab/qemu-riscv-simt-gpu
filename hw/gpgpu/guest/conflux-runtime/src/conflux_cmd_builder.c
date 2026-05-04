#include "conflux_cmd_builder.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- 创建 ---- */
conflux_cmd_builder_t *conflux_cmd_builder_create(conflux_queue_t *queue,
                                                 conflux_allocator_t *allocator) 
{
    if (!queue || !allocator) return NULL;
    
    conflux_cmd_builder_t *builder = malloc(sizeof(conflux_cmd_builder_t));
    if (!builder) return NULL;
    
    memset(builder, 0, sizeof(conflux_cmd_builder_t));
    builder->queue     = queue;
    builder->allocator = allocator;
    
    CONFLUX_INFO("[BUILDER] Created, queue=%p, allocator=%p\n",
                (void *)queue, (void *)allocator);
    
    return builder;
}

void conflux_cmd_builder_destroy(conflux_cmd_builder_t *builder) 
{
    if (!builder) return;
    
    CONFLUX_INFO("[BUILDER] Destroying: copies=%u, kernels=%u, allocs=%u, frees=%u\n",
                builder->copy_cmds_built,
                builder->kernel_cmds_built,
                builder->alloc_cmds_built,
                builder->free_cmds_built);
    
    free(builder);
}

/* ---- 内存拷贝 ---- */
int conflux_cmd_builder_copy(conflux_cmd_builder_t *builder,
                            const conflux_copy_request_t *req,
                            conflux_event_t **event_out) 
{
    if (!builder || !req) return CONFLUX_ERR_INVALID;
    if (req->size == 0) return CONFLUX_ERR_INVALID;
    
    /* 打包命令 */
    conflux_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type     = CONFLUX_CMD_COPY;
    cmd.src_addr = req->src_addr;
    cmd.dst_addr = req->dst_addr;
    cmd.size     = (uint32_t)req->size;
    
    /* 提交 */
    int ret = conflux_queue_submit(builder->queue, &cmd, event_out);
    if (ret == CONFLUX_SUCCESS) {
        builder->copy_cmds_built++;
    }
    
    return ret;
}

/* ---- 内核执行 ---- */
int conflux_cmd_builder_launch(conflux_cmd_builder_t *builder,
                              const conflux_launch_request_t *req,
                              conflux_event_t **event_out) 
{
    if (!builder || !req || !req->kernel) return CONFLUX_ERR_INVALID;
    
    conflux_kernel_t *k = req->kernel;
    
    /* 检查指令已上传 */
    if (!k->binary_uploaded) {
        CONFLUX_ERROR("[BUILDER] Kernel \"%s\" not uploaded\n", k->name);
        return CONFLUX_ERR_INVALID;
    }
    
    /* 把工作维度写入 kernel（临时覆盖） */
    k->work_dim = req->work_dim;
    for (int i = 0; i < 3; i++) {
        k->global_size[i] = req->global_size[i];
        k->local_size[i] = req->local_size[i];
    }
    
    /* 打包 */
    conflux_cmd_t cmd;
    int ret = conflux_kernel_pack_cmd(k, &cmd);
    if (ret != CONFLUX_SUCCESS) return ret;
    
    /* 提交 */
    ret = conflux_queue_submit(builder->queue, &cmd, event_out);
    if (ret == CONFLUX_SUCCESS) {
        builder->kernel_cmds_built++;
    }
    
    return ret;
}

/* ---- 显存分配 ---- */
int conflux_cmd_builder_alloc(conflux_cmd_builder_t *builder,
                             size_t size,
                             uint64_t *dev_addr_out,
                             conflux_event_t **event_out) 
{
    if (!builder || !dev_addr_out) return CONFLUX_ERR_INVALID;
    
    /* 用分配器分配地址 */
    uint64_t addr = conflux_allocator_alloc(builder->allocator, size);
    if (addr == UINT64_MAX) {
        return CONFLUX_ERR_MEM_OUT_OF_DEVICE;
    }
    
    *dev_addr_out = addr;
    
    /* 打包 ALLOC 命令 */
    conflux_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type     = CONFLUX_CMD_ALLOC;
    cmd.src_addr = addr;
    cmd.size     = (uint32_t)size;
    
    /* 提交 */
    int ret = conflux_queue_submit(builder->queue, &cmd, event_out);
    if (ret == CONFLUX_SUCCESS) {
        builder->alloc_cmds_built++;
    }
    
    return ret;
}

/* ---- 显存释放 ---- */
int conflux_cmd_builder_free(conflux_cmd_builder_t *builder,
                            uint64_t dev_addr,
                            size_t size,
                            conflux_event_t **event_out) 
{
    if (!builder) return CONFLUX_ERR_INVALID;
    
    /* 打包 FREE 命令 */
    conflux_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type     = CONFLUX_CMD_FREE;
    cmd.src_addr = dev_addr;
    cmd.size     = (uint32_t)size;
    
    int ret = conflux_queue_submit(builder->queue, &cmd, event_out);
    if (ret == CONFLUX_SUCCESS) {
        builder->free_cmds_built++;
    }
    
    /* 释放分配器记录 */
    conflux_allocator_free(builder->allocator, dev_addr, size);
    
    return ret;
}

/* ---- 屏障 ---- */
int conflux_cmd_builder_barrier(conflux_cmd_builder_t *builder,
                               conflux_event_t **event_out) 
{
    if (!builder) return CONFLUX_ERR_INVALID;
    
    conflux_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type = CONFLUX_CMD_BARRIER;
    
    return conflux_queue_submit(builder->queue, &cmd, event_out);
}

/* ---- 上传内核指令 ---- */
int conflux_cmd_builder_upload_kernel(conflux_cmd_builder_t *builder,
                                     conflux_kernel_t *kernel,
                                     uint64_t *dev_addr_out) 
{
    if (!builder || !kernel || !kernel->binary) return CONFLUX_ERR_INVALID;
    
    /* 分配一段设备内存存放指令 */
    uint64_t instr_addr = conflux_allocator_alloc(builder->allocator,
                                                  kernel->binary_size);
    if (instr_addr == UINT64_MAX) {
        return CONFLUX_ERR_MEM_OUT_OF_DEVICE;
    }
    
    /* 把指令从主机拷贝到设备 */
    conflux_copy_request_t req = {
        .src_addr = 0,  /* 特殊值：表示主机内存 */
        .dst_addr = instr_addr,
        .size     = kernel->binary_size,
    };
    
    /* 提交拷贝命令 */
    int ret = conflux_cmd_builder_copy(builder, &req, NULL);
    if (ret != CONFLUX_SUCCESS) {
        conflux_allocator_free(builder->allocator, instr_addr, 
                              kernel->binary_size);
        return ret;
    }
    
    /* 设置内核的设备地址 */
    kernel->binary_device_addr = instr_addr;
    kernel->binary_uploaded = 1;
    
    if (dev_addr_out) *dev_addr_out = instr_addr;
    
    CONFLUX_INFO("[BUILDER] Kernel \"%s\" uploaded to device addr 0x%lx\n",
                kernel->name, (unsigned long)instr_addr);
    
    return CONFLUX_SUCCESS;
}

/* ---- 调试 ---- */
/* dump 函数的 printf 全部保留，不做替换 */
void conflux_cmd_builder_dump(const conflux_cmd_builder_t *builder) 
{
    if (!builder) {
        printf("NULL builder\n");
        return;
    }
    
    printf("\n=== Command Builder ===\n");
    printf("  queue:      %p\n", (void *)builder->queue);
    printf("  allocator:  %p\n", (void *)builder->allocator);
    printf("  commands built:\n");
    printf("    copy:    %u\n", builder->copy_cmds_built);
    printf("    kernel:  %u\n", builder->kernel_cmds_built);
    printf("    alloc:   %u\n", builder->alloc_cmds_built);
    printf("    free:    %u\n", builder->free_cmds_built);
}