#include "conflux_kernel.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- 创建内核对象 ---- */
conflux_kernel_t *conflux_kernel_create(const char *name,
                                       uint32_t kernel_id,
                                       void *binary,
                                       size_t binary_size) 
{
    conflux_kernel_t *kernel = malloc(sizeof(conflux_kernel_t));
    if (!kernel) return NULL;
    
    memset(kernel, 0, sizeof(conflux_kernel_t));
    
    /* 名称（截断到 255 字符） */
    if (name) {
        strncpy(kernel->name, name, sizeof(kernel->name) - 1);
        kernel->name[sizeof(kernel->name) - 1] = '\0';
    }
    
    kernel->kernel_id   = kernel_id;
    kernel->binary      = binary;
    kernel->binary_size = binary_size;
    kernel->work_dim    = 1;  /* 默认 1D */
    
    /* 默认 global_size = 1, local_size = 0（让设备决定） */
    kernel->global_size[0] = 1;
    kernel->global_size[1] = 1;
    kernel->global_size[2] = 1;
    kernel->local_size[0] = 0;
    kernel->local_size[1] = 0;
    kernel->local_size[2] = 0;
    
    CONFLUX_INFO("[KERNEL] Created \"%s\" id=%u binary_size=%zu\n",
                kernel->name, kernel_id, binary_size);
    
    return kernel;
}

/* ---- 销毁 ---- */
void conflux_kernel_destroy(conflux_kernel_t *kernel) 
{
    if (!kernel) return;
    
    CONFLUX_INFO("[KERNEL] Destroying \"%s\"\n", kernel->name);
    free(kernel);
}

/* ---- 上传指令到设备 ---- */
int conflux_kernel_upload(conflux_kernel_t *kernel, uint64_t device_addr) 
{
    if (!kernel) return CONFLUX_ERR_INVALID;
    if (!kernel->binary) {
        CONFLUX_ERROR("[KERNEL] \"%s\" has no binary to upload\n", kernel->name);
        return CONFLUX_ERR_INVALID;
    }
    
    kernel->binary_device_addr = device_addr;
    kernel->binary_uploaded = 1;
    
    CONFLUX_INFO("[KERNEL] \"%s\" binary uploaded: host=%p → device=0x%lx (%zu bytes)\n",
                kernel->name, kernel->binary, 
                (unsigned long)device_addr, kernel->binary_size);
    
    return CONFLUX_SUCCESS;
}

/* ---- 设置参数 ---- */
int conflux_kernel_set_arg(conflux_kernel_t *kernel,
                          uint32_t arg_index,
                          size_t arg_size,
                          const void *arg_value,
                          int is_local) 
{
    if (!kernel) return CONFLUX_ERR_INVALID;
    
    if (arg_index >= CONFLUX_KERNEL_MAX_ARGS) {
        CONFLUX_ERROR("[KERNEL] \"%s\" arg index %u exceeds max %u\n",
                     kernel->name, arg_index, CONFLUX_KERNEL_MAX_ARGS);
        return CONFLUX_ERR_INVALID;
    }
    
    conflux_kernel_arg_t *arg = &kernel->args[arg_index];
    
    arg->size     = arg_size;
    arg->value    = (void *)arg_value;  /* 存指针 */
    arg->is_local = is_local;
    
    /* 更新已设置参数计数 */
    if (arg_index >= kernel->num_args) {
        kernel->num_args = arg_index + 1;
    }
    
    CONFLUX_DEBUG("[KERNEL] \"%s\" arg[%u] set: size=%zu, is_local=%d\n",
                 kernel->name, arg_index, arg_size, is_local);
    
    return CONFLUX_SUCCESS;
}

/* ---- 设置工作维度 ---- */
int conflux_kernel_set_work_dim(conflux_kernel_t *kernel, uint32_t work_dim) 
{
    if (!kernel) return CONFLUX_ERR_INVALID;
    if (work_dim < 1 || work_dim > 3) return CONFLUX_ERR_INVALID;
    
    kernel->work_dim = work_dim;
    return CONFLUX_SUCCESS;
}

int conflux_kernel_set_global_size(conflux_kernel_t *kernel,
                                   uint32_t dim, size_t size) 
{
    if (!kernel || dim >= 3) return CONFLUX_ERR_INVALID;
    kernel->global_size[dim] = size;
    return CONFLUX_SUCCESS;
}

int conflux_kernel_set_local_size(conflux_kernel_t *kernel,
                                  uint32_t dim, size_t size) 
{
    if (!kernel || dim >= 3) return CONFLUX_ERR_INVALID;
    kernel->local_size[dim] = size;
    return CONFLUX_SUCCESS;
}

/* ---- 关联队列 ---- */
void conflux_kernel_set_queue(conflux_kernel_t *kernel, void *queue) 
{
    if (!kernel) return;
    kernel->queue = queue;
    CONFLUX_DEBUG("[KERNEL] \"%s\" bound to queue %p\n", kernel->name, queue);
}

/* ---- 打包成命令描述符 ---- */
int conflux_kernel_pack_cmd(const conflux_kernel_t *kernel,
                            conflux_cmd_t *cmd_out) 
{
    if (!kernel || !cmd_out) return CONFLUX_ERR_INVALID;
    
    /* 检查必要设置 */
    if (!kernel->binary_uploaded) {
        CONFLUX_ERROR("[KERNEL] \"%s\" binary not uploaded yet\n", kernel->name);
        return CONFLUX_ERR_INVALID;
    }
    
    if (kernel->num_args == 0) {
        CONFLUX_WARN("[KERNEL] \"%s\" warning: no arguments set\n", kernel->name);
    }
    
    memset(cmd_out, 0, sizeof(conflux_cmd_t));
    
    cmd_out->type      = CONFLUX_CMD_KERNEL;
    cmd_out->src_addr  = kernel->binary_device_addr;  /* 指令地址 */
    cmd_out->kernel_id = kernel->kernel_id;
    
    /* 
     * 把工作维度信息编码进 flags
     * bits [1:0] = work_dim (1-3)
     * bits [31:2] = 保留
     */
    cmd_out->flags = (kernel->work_dim & 0x3);
    
    /*
     * 用 size 字段传 global_size[0]（主要工作项数）
     * 其他维度信息在实际系统中通过参数缓冲区传
     */
    cmd_out->size = (uint32_t)kernel->global_size[0];
    
    CONFLUX_DEBUG("[KERNEL] \"%s\" packed cmd: type=KERNEL, instr_addr=0x%lx, "
                 "dim=%u, global=%zu\n",
                 kernel->name, (unsigned long)cmd_out->src_addr,
                 kernel->work_dim, kernel->global_size[0]);
    
    return CONFLUX_SUCCESS;
}

/* ---- 调试 dump，完全保留原始 printf ---- */
void conflux_kernel_dump(const conflux_kernel_t *kernel) 
{
    if (!kernel) {
        printf("NULL kernel\n");
        return;
    }
    
    printf("\n=== Kernel: %s ===\n", kernel->name);
    printf("  id:            %u\n", kernel->kernel_id);
    printf("  binary:        host=%p, device=0x%lx, size=%zu, uploaded=%d\n",
           kernel->binary,
           (unsigned long)kernel->binary_device_addr,
           kernel->binary_size,
           kernel->binary_uploaded);
    printf("  work_dim:      %u\n", kernel->work_dim);
    printf("  global_size:   %zu × %zu × %zu\n",
           kernel->global_size[0], kernel->global_size[1], kernel->global_size[2]);
    printf("  local_size:    %zu × %zu × %zu\n",
           kernel->local_size[0], kernel->local_size[1], kernel->local_size[2]);
    printf("  num_args:      %u\n", kernel->num_args);
    printf("  queue:         %p\n", kernel->queue);
    
    if (kernel->num_args > 0) {
        printf("  Arguments:\n");
        for (uint32_t i = 0; i < kernel->num_args; i++) {
            const conflux_kernel_arg_t *arg = &kernel->args[i];
            printf("    [%u] size=%zu, local=%d, value=%p\n",
                   i, arg->size, arg->is_local, arg->value);
        }
    }
}