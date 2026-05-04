#include "conflux_device.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* ---- 创建 ---- */
conflux_device_t *conflux_device_create(void) 
{
    conflux_device_t *dev = malloc(sizeof(conflux_device_t));
    if (!dev) return NULL;
    
    memset(dev, 0, sizeof(conflux_device_t));
    
    /* 设置默认属性 */
    strncpy(dev->name, CONFLUX_DEVICE_NAME, sizeof(dev->name) - 1);
    strncpy(dev->vendor, CONFLUX_DEVICE_VENDOR, sizeof(dev->vendor) - 1);
    dev->vendor_id    = CONFLUX_DEVICE_VENDOR_ID;
    dev->device_id    = CONFLUX_DEVICE_DEVICE_ID;
    
    dev->max_compute_units = CONFLUX_DEFAULT_COMPUTE_UNITS;
    dev->max_work_item_dims = CONFLUX_DEFAULT_MAX_WORK_DIMS;
    
    /* 默认每维可处理大量工作项 */
    dev->max_work_item_sizes[0] = 1024;
    dev->max_work_item_sizes[1] = 1024;
    dev->max_work_item_sizes[2] = 64;
    dev->max_work_group_size    = CONFLUX_DEFAULT_MAX_GROUP_SIZE;
    dev->max_clock_frequency    = CONFLUX_DEFAULT_CLOCK_FREQ;
    dev->address_bits           = CONFLUX_DEFAULT_ADDR_BITS;
    
    dev->global_mem_size   = CONFLUX_DEFAULT_GLOBAL_MEM;
    dev->local_mem_size    = CONFLUX_DEFAULT_LOCAL_MEM;
    dev->max_mem_alloc_size = CONFLUX_DEFAULT_MAX_ALLOC;
    dev->mem_block_size    = 4 * 1024;  /* 4KB */
    
    dev->max_queues = CONFLUX_DEVICE_MAX_QUEUES;
    dev->num_queues = 0;

    /* HAL 暂未初始化，conflux_device_init 中再做 */
    memset(&dev->hal, 0, sizeof(dev->hal));

    pthread_mutex_init(&dev->lock, NULL);
    
    CONFLUX_INFO("[DEVICE] Created (not yet initialized)\n");
    
    return dev;
}

/* ---- 销毁 ---- */
void conflux_device_destroy(conflux_device_t *dev) 
{
    if (!dev) return;
    
    CONFLUX_INFO("[DEVICE] Destroying...\n");
    
    /* 如果设备在线，先离线 */
    if (dev->flags & CONFLUX_DEV_FLAG_ONLINE) {
        conflux_device_offline(dev);
    }

    /* 销毁所有惰性创建的队列 */
    for (uint32_t i = 0; i < dev->max_queues; i++) {
        if (dev->queues[i]) {
            conflux_queue_stop_consumer(dev->queues[i]);
            conflux_queue_destroy(dev->queues[i]);
            dev->queues[i] = NULL;
        }
    }
    dev->num_queues = 0;

    /* 如果分配器已初始化，销毁 */
    if (dev->flags & CONFLUX_DEV_FLAG_INITIALIZED) {
        conflux_allocator_destroy(&dev->allocator);
    }

    /* 关闭 HAL（会负责 close fd / munmap 等） */
    conflux_hal_close(&dev->hal);

    pthread_mutex_destroy(&dev->lock);
    
    free(dev);
}

/* ---- 初始化（模拟打开设备） ---- */
int conflux_device_init(conflux_device_t *dev,
                       const char *dev_path,
                       uint64_t mmio_base,
                       uint64_t mmio_size) 
{
    if (!dev) return CONFLUX_ERR_INVALID;
    
    pthread_mutex_lock(&dev->lock);

    /* 通过 HAL 打开设备：有非空路径走 IOCTL，否则走 SIM 模式 */
    bool has_path = (dev_path != NULL) && (dev_path[0] != '\0');
    conflux_hal_mode_t mode = has_path ? CONFLUX_HAL_MODE_IOCTL
                                       : CONFLUX_HAL_MODE_SIM;
    int hal_ret = conflux_hal_init(&dev->hal, mode,
                                   has_path ? dev_path : NULL);
    if (hal_ret != CONFLUX_SUCCESS) {
        CONFLUX_ERROR("[DEVICE] HAL init failed (mode=%d, path=%s)\n",
                     mode, dev_path ? dev_path : "(sim)");
        dev->last_error = hal_ret;
        pthread_mutex_unlock(&dev->lock);
        return hal_ret;
    }
    CONFLUX_INFO("[DEVICE] HAL ready (mode=%s)\n",
                mode == CONFLUX_HAL_MODE_IOCTL ? "ioctl" : "sim");

    dev->mmio_base = mmio_base;
    dev->mmio_size = mmio_size;

    /* 初始化显存分配器
     * DMA ioctl 的 dst_addr 是 VRAM 内偏移（从 0 开始），不是物理地址 */
    int ret = conflux_allocator_init(&dev->allocator,
                                     0,                 /* VRAM 偏移从 0 起 */
                                     dev->global_mem_size,
                                     dev->mem_block_size);
    if (ret != 0) {
        CONFLUX_ERROR("[DEVICE] Allocator init failed\n");
        dev->last_error = CONFLUX_ERR_NOMEM;
        conflux_hal_close(&dev->hal);
        pthread_mutex_unlock(&dev->lock);
        return CONFLUX_ERR_NOMEM;
    }
    
    dev->flags |= CONFLUX_DEV_FLAG_INITIALIZED;
    dev->last_error = CONFLUX_SUCCESS;
    
    pthread_mutex_unlock(&dev->lock);
    
    CONFLUX_INFO("[DEVICE] Initialized: %s, %lu MB global, %lu KB local\n",
                dev->name,
                (unsigned long)(dev->global_mem_size / (1024 * 1024)),
                (unsigned long)(dev->local_mem_size / 1024));
    
    return CONFLUX_SUCCESS;
}

/* ---- 在线/离线 ---- */
int conflux_device_online(conflux_device_t *dev) 
{
    if (!dev) return CONFLUX_ERR_INVALID;
    
    if (!(dev->flags & CONFLUX_DEV_FLAG_INITIALIZED)) {
        return CONFLUX_ERR_DEVICE_NOT_READY;
    }
    
    pthread_mutex_lock(&dev->lock);
    dev->flags |= CONFLUX_DEV_FLAG_ONLINE;
    dev->last_error = CONFLUX_SUCCESS;
    pthread_mutex_unlock(&dev->lock);
    
    CONFLUX_INFO("[DEVICE] Now online\n");
    return CONFLUX_SUCCESS;
}

int conflux_device_offline(conflux_device_t *dev) 
{
    if (!dev) return CONFLUX_ERR_INVALID;
    
    pthread_mutex_lock(&dev->lock);
    dev->flags &= ~CONFLUX_DEV_FLAG_ONLINE;
    pthread_mutex_unlock(&dev->lock);
    
    CONFLUX_INFO("[DEVICE] Now offline\n");
    return CONFLUX_SUCCESS;
}

int conflux_device_is_online(const conflux_device_t *dev) 
{
    if (!dev) return 0;
    return (dev->flags & CONFLUX_DEV_FLAG_ONLINE) != 0;
}

/* ---- 复位 ---- */
int conflux_device_reset(conflux_device_t *dev) 
{
    if (!dev) return CONFLUX_ERR_INVALID;
    
    CONFLUX_INFO("[DEVICE] Resetting...\n");
    
    pthread_mutex_lock(&dev->lock);
    
    /* 重新初始化分配器 */
    conflux_allocator_destroy(&dev->allocator);
    int ret = conflux_allocator_init(&dev->allocator,
                                     0,
                                     dev->global_mem_size,
                                     dev->mem_block_size);
    if (ret != 0) {
        dev->flags |= CONFLUX_DEV_FLAG_ERROR;
        dev->last_error = CONFLUX_ERR_DEVICE_RESET_FAIL;
        pthread_mutex_unlock(&dev->lock);
        return CONFLUX_ERR_DEVICE_RESET_FAIL;
    }
    
    dev->flags &= ~CONFLUX_DEV_FLAG_ERROR;
    dev->last_error = CONFLUX_SUCCESS;
    dev->num_queues = 0;
    
    pthread_mutex_unlock(&dev->lock);
    
    CONFLUX_INFO("[DEVICE] Reset complete\n");
    return CONFLUX_SUCCESS;
}

/* ---- 查询属性 ---- */
void conflux_device_query_info(const conflux_device_t *dev,
                              char *buf, size_t buf_size) 
{
    if (!dev || !buf) return;
    
    snprintf(buf, buf_size,
             "Device: %s\n"
             "  Vendor:       %s (0x%04X)\n"
             "  Device ID:    0x%04X\n"
             "  Compute Units: %u\n"
             "  Max Work Dims: %u\n"
             "  Max Group Size:%zu\n"
             "  Max Work Items:[%zu, %zu, %zu]\n"
             "  Clock Freq:    %u MHz\n"
             "  Address Bits:  %u\n"
             "  Global Memory: %lu MB\n"
             "  Local Memory:  %lu KB\n"
             "  Max Alloc:     %lu MB\n"
             "  Queues:        %u/%u\n"
             "  Online:        %s\n"
             "  HAL mode:      %d  fd=%d\n",
             dev->name,
             dev->vendor, dev->vendor_id,
             dev->device_id,
             dev->max_compute_units,
             dev->max_work_item_dims,
             dev->max_work_group_size,
             dev->max_work_item_sizes[0],
             dev->max_work_item_sizes[1],
             dev->max_work_item_sizes[2],
             dev->max_clock_frequency,
             dev->address_bits,
             (unsigned long)(dev->global_mem_size / (1024 * 1024)),
             (unsigned long)(dev->local_mem_size / 1024),
             (unsigned long)(dev->max_mem_alloc_size / (1024 * 1024)),
             dev->num_queues, dev->max_queues,
             conflux_device_is_online(dev) ? "yes" : "no",
             dev->hal.mode, dev->hal.dev_fd);
}

/* ---- 获取分配器 ---- */
conflux_allocator_t *conflux_device_get_allocator(conflux_device_t *dev)
{
    if (!dev || !(dev->flags & CONFLUX_DEV_FLAG_INITIALIZED)) {
        return NULL;
    }
    return &dev->allocator;
}

/* ---- 队列消费者：把 conflux_cmd_t 翻译成 HAL 调用 ----
 *
 * 在 conflux_queue 的消费者线程中被调用。同步执行，返回 0 表示成功。
 */
static int device_execute_cmd(conflux_cmd_t *cmd, void *user_data)
{
    conflux_device_t *dev = (conflux_device_t *)user_data;
    if (!dev || !cmd) return -1;

    switch (cmd->type) {
        case CONFLUX_CMD_NOP:
            return 0;

        case CONFLUX_CMD_COPY: {
            /* 通用 device-to-device 拷贝：通过 DMA */
            int r = conflux_hal_dma_transfer(&dev->hal,
                                             cmd->src_addr,
                                             cmd->dst_addr,
                                             cmd->size,
                                             /*to_device*/ true);
            return (r == CONFLUX_SUCCESS) ? 0 : -1;
        }

        case CONFLUX_CMD_KERNEL: {
            /* cmd 中只携带 kernel_id；真正的 grid/block 由上层在
             * 提交前设置好寄存器，这里只触发 dispatch + wait */
            int r = conflux_hal_dispatch(&dev->hal);
            if (r != CONFLUX_SUCCESS) return -1;
            r = conflux_hal_wait_kernel(&dev->hal, /*timeout*/ 0);
            return (r == CONFLUX_SUCCESS) ? 0 : -1;
        }

        case CONFLUX_CMD_BARRIER:
            return conflux_hal_wait_kernel(&dev->hal, 0) == CONFLUX_SUCCESS ? 0 : -1;

        case CONFLUX_CMD_ALLOC:
        case CONFLUX_CMD_FREE:
            /* 分配/释放在主机侧 allocator 完成，队列里不应该出现 */
            CONFLUX_WARN("[DEVICE] alloc/free should not flow through queue");
            return 0;

        default:
            CONFLUX_WARN("[DEVICE] unknown cmd type %d", cmd->type);
            return -1;
    }
}

/* ---- 获取/惰性创建命令队列 ---- */
conflux_queue_t *conflux_device_get_queue(conflux_device_t *dev,
                                          uint32_t index)
{
    if (!dev || index >= dev->max_queues) return NULL;
    if (!(dev->flags & CONFLUX_DEV_FLAG_INITIALIZED)) return NULL;

    pthread_mutex_lock(&dev->lock);

    if (dev->queues[index] == NULL) {
        conflux_queue_t *q = conflux_queue_create(/*ring_size*/ 64,
                                                  device_execute_cmd,
                                                  dev);
        if (!q) {
            pthread_mutex_unlock(&dev->lock);
            CONFLUX_ERROR("[DEVICE] queue %u: create failed", index);
            return NULL;
        }
        if (conflux_queue_start_consumer(q) != CONFLUX_SUCCESS) {
            conflux_queue_destroy(q);
            pthread_mutex_unlock(&dev->lock);
            CONFLUX_ERROR("[DEVICE] queue %u: start_consumer failed", index);
            return NULL;
        }
        dev->queues[index] = q;
        if (index >= dev->num_queues) dev->num_queues = index + 1;
        CONFLUX_INFO("[DEVICE] queue %u created (ring=64)", index);
    }

    conflux_queue_t *result = dev->queues[index];
    pthread_mutex_unlock(&dev->lock);
    return result;
}

/* ---- 错误码 ---- */
int conflux_device_get_last_error(const conflux_device_t *dev) 
{
    if (!dev) return CONFLUX_ERR_INVALID;
    return dev->last_error;
}

/* ---- 调试 ---- */
/* dump 函数的 printf 全部保留 */
void conflux_device_dump(const conflux_device_t *dev) 
{
    if (!dev) {
        printf("NULL device\n");
        return;
    }
    
    char info[1024];
    conflux_device_query_info(dev, info, sizeof(info));
    printf("\n%s\n", info);
    
    if (dev->flags & CONFLUX_DEV_FLAG_INITIALIZED) {
        conflux_allocator_dump(&dev->allocator);
    }
}