#ifndef CONFLUX_KERNEL_H
#define CONFLUX_KERNEL_H

#include "conflux_error.h"
#include "conflux_queue.h"   /* conflux_cmd_t */
#include <stdint.h>
#include <stddef.h>

/* 一个内核最多接受的参数数量 */
#define CONFLUX_KERNEL_MAX_ARGS 32

/* 参数信息 */
typedef struct {
    void   *value;        /* 参数值指针 */
    size_t  size;         /* 参数大小（字节） */
    int     is_local;     /* 1 = local memory, 0 = global/constant */
} conflux_kernel_arg_t;

/* 内核对象 */
typedef struct {
    /* 标识 */
    char     name[256];              /* 内核函数名 */
    uint32_t kernel_id;              /* 设备上的唯一 ID */
    
    /* 编译后的指令 */
    void    *binary;                 /* 主机端指针 */
    size_t   binary_size;            /* 指令大小（字节） */
    uint64_t binary_device_addr;     /* 指令在设备显存上的地址 */
    int      binary_uploaded;        /* 是否已上传到设备 */
    
    /* 参数 */
    uint32_t num_args;               /* 已设置的参数数量 */
    conflux_kernel_arg_t args[CONFLUX_KERNEL_MAX_ARGS];
    
    /* 工作维度（NDRange） */
    uint32_t work_dim;               /* 1, 2, 或 3 */
    size_t   global_size[3];         /* 全局工作项总数 */
    size_t   local_size[3];          /* 每个工作组的大小（0 = 自动） */
    
    /* 关联的命令队列（提交时用） */
    void *queue;                     /* conflux_queue_t * */
    
} conflux_kernel_t;

/* ---- API ---- */

/* 创建/销毁 */
conflux_kernel_t *conflux_kernel_create(const char *name,
                                       uint32_t kernel_id,
                                       void *binary,
                                       size_t binary_size);
void conflux_kernel_destroy(conflux_kernel_t *kernel);

/* 上传指令到设备 */
int conflux_kernel_upload(conflux_kernel_t *kernel, uint64_t device_addr);

/* 设置参数（模仿 clSetKernelArg） */
int conflux_kernel_set_arg(conflux_kernel_t *kernel,
                          uint32_t arg_index,
                          size_t arg_size,
                          const void *arg_value,
                          int is_local);

/* 设置工作维度 */
int conflux_kernel_set_work_dim(conflux_kernel_t *kernel,
                                uint32_t work_dim);
int conflux_kernel_set_global_size(conflux_kernel_t *kernel,
                                   uint32_t dim,
                                   size_t size);
int conflux_kernel_set_local_size(conflux_kernel_t *kernel,
                                  uint32_t dim,
                                  size_t size);

/* 关联命令队列 */
void conflux_kernel_set_queue(conflux_kernel_t *kernel, void *queue);

/* 打包成命令描述符（准备提交） */
int conflux_kernel_pack_cmd(const conflux_kernel_t *kernel,
                            conflux_cmd_t *cmd_out);

/* 调试 */
void conflux_kernel_dump(const conflux_kernel_t *kernel);

#endif