#ifndef CONFLUX_PLATFORM_H
#define CONFLUX_PLATFORM_H

#include "conflux_device.h"
#include "conflux_error.h"
#include <stdint.h>
#include <stddef.h>

/* 最大支持的设备数 */
#define CONFLUX_MAX_DEVICES 16

/* ---- 设备描述符（probe 时发现的信息） ---- */
typedef struct {
    int         dev_index;       /* 平台内编号 0..MAX-1 */
    char        path[256];       /* 设备路径（/dev/gpgpu0） */
    uint64_t    mmio_base;       /* MMIO 物理基地址 */
    uint64_t    mmio_size;       /* MMIO 大小 */
    uint32_t    vendor_id;       /* PCI vendor */
    uint32_t    device_id;       /* PCI device */
    int         available;       /* 是否可用 */
} conflux_device_desc_t;

/* ---- 平台状态 ---- */
typedef struct {
    int     initialized;         /* 是否已 probe */
    int     num_devices;         /* 已发现的设备数 */
    
    /* 设备描述符（probe 时填充） */
    conflux_device_desc_t descs[CONFLUX_MAX_DEVICES];
    
    /* 已初始化的设备对象（按需创建） */
    conflux_device_t *devices[CONFLUX_MAX_DEVICES];
    
    /* 默认设备索引 */
    int     default_device;
    
    /* 平台级锁 */
    pthread_mutex_t lock;
    
} conflux_platform_t;

/* ---- 平台 API ---- */

/* 获取全局单例 */
conflux_platform_t *conflux_platform_get(void);

/* 初始化/清理 */
int  conflux_platform_init(void);
void conflux_platform_destroy(void);

/* 设备发现（probe） */
int  conflux_platform_probe(void);
int  conflux_platform_probe_specific(const conflux_device_desc_t *desc);

/* 设备管理 */
int  conflux_platform_open_device(int dev_index);
void conflux_platform_close_device(int dev_index);
void conflux_platform_close_all(void);

/* 查询 */
int  conflux_platform_get_num_devices(void);
conflux_device_t *conflux_platform_get_device(int dev_index);
conflux_device_t *conflux_platform_get_default_device(void);
const conflux_device_desc_t *conflux_platform_get_desc(int dev_index);

/* 多设备路由（根据负载选择设备） */
int  conflux_platform_pick_device(void);

/* 调试 */
void conflux_platform_dump(void);

#endif