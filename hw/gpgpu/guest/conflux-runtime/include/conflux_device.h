#ifndef CONFLUX_DEVICE_H
#define CONFLUX_DEVICE_H

#include "conflux_error.h"
#include "conflux_allocator.h"
#include "conflux_queue.h"
#include "conflux_hal.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

/* ---- 设备属性常量（Conflux SimX 的规格） ---- */
#define CONFLUX_DEVICE_NAME           "Conflux GPGPU (SimX)"
#define CONFLUX_DEVICE_VENDOR         "Conflux"
#define CONFLUX_DEVICE_VENDOR_ID      0x1AF4  /* 虚拟 PCI vendor */
#define CONFLUX_DEVICE_DEVICE_ID      0x1050  /* 虚拟 PCI device */
#define CONFLUX_DEVICE_MAX_QUEUES     4

/* 默认设备规格（可被 init 覆盖） */
#define CONFLUX_DEFAULT_GLOBAL_MEM    (64ULL * 1024 * 1024)   /* 64 MB */
#define CONFLUX_DEFAULT_LOCAL_MEM     (64ULL * 1024)          /* 64 KB */
#define CONFLUX_DEFAULT_MAX_ALLOC     (32ULL * 1024 * 1024)   /* 32 MB */
#define CONFLUX_DEFAULT_COMPUTE_UNITS 4
#define CONFLUX_DEFAULT_MAX_WORK_DIMS 3
#define CONFLUX_DEFAULT_MAX_GROUP_SIZE 256
#define CONFLUX_DEFAULT_CLOCK_FREQ    1000                    /* MHz */
#define CONFLUX_DEFAULT_ADDR_BITS     64

/* ---- 设备上下文结构体 ---- */
typedef struct {
    /* ===== 硬件访问 ===== */
    conflux_hal_t hal;            /* 硬件抽象层（持有 fd / mmap / 模式） */
    uint64_t mmio_base;           /* 显存逻辑起始地址（给分配器用） */
    uint64_t mmio_size;           /* 显存大小 */
    
    /* ===== 设备属性 ===== */
    char   name[64];              /* 设备名称 */
    char   vendor[32];            /* 厂商名称 */
    uint32_t vendor_id;           /* PCI vendor ID */
    uint32_t device_id;           /* PCI device ID */
    
    /* 计算能力 */
    uint32_t max_compute_units;   /* 最大计算单元数 */
    uint32_t max_work_item_dims;  /* 最大工作维度（1/2/3） */
    size_t   max_work_item_sizes[3]; /* 每维最大工作项数 */
    size_t   max_work_group_size; /* 每组最大工作项 */
    uint32_t max_clock_frequency; /* 最大频率（MHz） */
    uint32_t address_bits;        /* 地址宽度（32 或 64） */
    
    /* 内存属性 */
    uint64_t global_mem_size;     /* 全局内存（显存）总大小 */
    uint64_t local_mem_size;      /* 局部内存总大小 */
    uint64_t max_mem_alloc_size;  /* 单次最大分配 */
    uint32_t mem_block_size;      /* 位图分配器的块大小 */
    
    /* ===== 子系统 ===== */
    conflux_allocator_t allocator; /* 显存分配器 */
    
    /* 命令队列（惰性创建，由 conflux_device_get_queue 管理） */
    conflux_queue_t *queues[CONFLUX_DEVICE_MAX_QUEUES];
    uint32_t num_queues;          /* 当前创建的队列数 */
    uint32_t max_queues;          /* 最大队列数 */
    
    /* ===== 运行时状态 ===== */
    volatile uint32_t flags;      /* 设备状态标志 */
#define CONFLUX_DEV_FLAG_INITIALIZED   (1 << 0)
#define CONFLUX_DEV_FLAG_ONLINE        (1 << 1)
#define CONFLUX_DEV_FLAG_ERROR         (1 << 2)
    
    int last_error;               /* 最近一次错误码 */
    
    pthread_mutex_t lock;         /* 保护设备级操作 */
    
} conflux_device_t;

/* ---- 设备状态标志 ---- */
#define CONFLUX_DEV_STATUS_IDLE    0
#define CONFLUX_DEV_STATUS_BUSY    1
#define CONFLUX_DEV_STATUS_ERROR   2

/* ---- API ---- */

/* 创建/销毁 */
conflux_device_t *conflux_device_create(void);
void conflux_device_destroy(conflux_device_t *dev);

/* 初始化（模拟打开设备） */
int conflux_device_init(conflux_device_t *dev,
                       const char *dev_path,
                       uint64_t mmio_base,
                       uint64_t mmio_size);

/* 在线/离线 */
int conflux_device_online(conflux_device_t *dev);
int conflux_device_offline(conflux_device_t *dev);
int conflux_device_is_online(const conflux_device_t *dev);

/* 复位 */
int conflux_device_reset(conflux_device_t *dev);

/* 查询属性 */
void conflux_device_query_info(const conflux_device_t *dev,
                              char *buf, size_t buf_size);

/* 获取分配器（给 builder 用） */
conflux_allocator_t *conflux_device_get_allocator(conflux_device_t *dev);

/* 获取命令队列（按索引；首次访问时惰性创建） */
conflux_queue_t *conflux_device_get_queue(conflux_device_t *dev,
                                          uint32_t index);

/* 获取错误码 */
int conflux_device_get_last_error(const conflux_device_t *dev);
const char *conflux_device_strerror(int error);

/* 调试 */
void conflux_device_dump(const conflux_device_t *dev);

#endif