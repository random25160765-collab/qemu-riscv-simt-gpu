#include "conflux_platform.h"
#include "conflux_log.h"          /* 新增日志头文件 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>   /* access() */

/* ---- 全局单例 ---- */
static conflux_platform_t g_platform = {0};

conflux_platform_t *conflux_platform_get(void) 
{
    return &g_platform;
}

/* ---- 初始化 ---- */
int conflux_platform_init(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (plat->initialized) {
        CONFLUX_INFO("[PLATFORM] Already initialized\n");
        return CONFLUX_SUCCESS;
    }
    
    memset(plat, 0, sizeof(conflux_platform_t));
    plat->default_device = -1;
    pthread_mutex_init(&plat->lock, NULL);
    
    plat->initialized = 1;
    
    CONFLUX_INFO("[PLATFORM] Initialized (max %d devices)\n", CONFLUX_MAX_DEVICES);
    
    return CONFLUX_SUCCESS;
}

/* ---- 清理 ---- */
void conflux_platform_destroy(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) return;
    
    CONFLUX_INFO("[PLATFORM] Destroying...\n");
    
    /* 关闭所有设备 */
    conflux_platform_close_all();
    
    plat->initialized = 0;
    plat->num_devices = 0;
    
    pthread_mutex_destroy(&plat->lock);
    
    CONFLUX_INFO("[PLATFORM] Destroyed\n");
}

/* ---- 设备发现（默认：探测模拟设备） ---- */
int conflux_platform_probe(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) {
        int ret = conflux_platform_init();
        if (ret != CONFLUX_SUCCESS) return ret;
    }
    
    pthread_mutex_lock(&plat->lock);
    
    /* 
     * 真实环境：扫描 /dev/gpgpu* 或 PCI 总线。
     * 这里模拟：创建 1 个默认设备。
     */
    /* 严格生产模式：必须存在 /dev/gpgpu0（先 make install 内核驱动）
     * SIM 模式不在 probe 路径里 — 测试代码请用 probe_specific 显式注入。 */
    if (access("/dev/gpgpu0", F_OK) != 0) {
        CONFLUX_ERROR("[PLATFORM] /dev/gpgpu0 not found "
                     "(run 'make install' in guest/driver to load gpgpu.ko)");
        plat->num_devices = 0;
        plat->default_device = -1;
        pthread_mutex_unlock(&plat->lock);
        return 0;
    }

    plat->num_devices = 1;

    conflux_device_desc_t *desc = &plat->descs[0];
    memset(desc, 0, sizeof(conflux_device_desc_t));
    desc->dev_index  = 0;
    desc->vendor_id  = CONFLUX_DEVICE_VENDOR_ID;
    desc->device_id  = CONFLUX_DEVICE_DEVICE_ID;
    desc->mmio_base  = 0x10000000;     /* 256 MB */
    desc->mmio_size  = 64 * 1024 * 1024;  /* 64 MB MMIO */
    desc->available  = 1;
    snprintf(desc->path, sizeof(desc->path), "/dev/gpgpu0");
    
    plat->default_device = 0;
    
    pthread_mutex_unlock(&plat->lock);
    
    CONFLUX_INFO("[PLATFORM] Probe found %d device(s)\n", plat->num_devices);
    
    return plat->num_devices;
}

/* ---- 添加指定设备（用于开多个 SimX 实例） ---- */
int conflux_platform_probe_specific(const conflux_device_desc_t *desc) 
{
    if (!desc) return CONFLUX_ERR_INVALID;
    
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) {
        int ret = conflux_platform_init();
        if (ret != CONFLUX_SUCCESS) return ret;
    }
    
    pthread_mutex_lock(&plat->lock);
    
    if (plat->num_devices >= CONFLUX_MAX_DEVICES) {
        CONFLUX_ERROR("[PLATFORM] Cannot add device: max %d reached\n",
                     CONFLUX_MAX_DEVICES);
        pthread_mutex_unlock(&plat->lock);
        return CONFLUX_ERR_NOMEM;
    }
    
    int idx = plat->num_devices;
    memcpy(&plat->descs[idx], desc, sizeof(conflux_device_desc_t));
    plat->descs[idx].dev_index = idx;
    plat->num_devices++;
    
    if (plat->default_device < 0) {
        plat->default_device = 0;
    }
    
    pthread_mutex_unlock(&plat->lock);
    
    CONFLUX_INFO("[PLATFORM] Added device %d: %s\n", idx, desc->path);
    
    return idx;
}

/* ---- 打开设备 ---- */
int conflux_platform_open_device(int dev_index) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) return CONFLUX_ERR_DEVICE_NOT_READY;
    if (dev_index < 0 || dev_index >= plat->num_devices) {
        return CONFLUX_ERR_INVALID;
    }
    
    pthread_mutex_lock(&plat->lock);
    
    /* 已经打开就返回 */
    if (plat->devices[dev_index] != NULL) {
        CONFLUX_INFO("[PLATFORM] Device %d already open\n", dev_index);
        pthread_mutex_unlock(&plat->lock);
        return CONFLUX_SUCCESS;
    }
    
    /* 创建设备对象 */
    conflux_device_t *dev = conflux_device_create();
    if (!dev) {
        pthread_mutex_unlock(&plat->lock);
        return CONFLUX_ERR_NOMEM;
    }
    
    /* 初始化 */
    conflux_device_desc_t *desc = &plat->descs[dev_index];
    int ret = conflux_device_init(dev, desc->path,
                                  desc->mmio_base, desc->mmio_size);
    if (ret != CONFLUX_SUCCESS) {
        conflux_device_destroy(dev);
        pthread_mutex_unlock(&plat->lock);
        return ret;
    }
    
    /* 上线 */
    conflux_device_online(dev);
    
    plat->devices[dev_index] = dev;
    
    pthread_mutex_unlock(&plat->lock);
    
    CONFLUX_INFO("[PLATFORM] Opened device %d: %s\n", dev_index, dev->name);
    
    return CONFLUX_SUCCESS;
}

/* ---- 关闭设备 ---- */
void conflux_platform_close_device(int dev_index) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) return;
    if (dev_index < 0 || dev_index >= plat->num_devices) return;
    
    pthread_mutex_lock(&plat->lock);
    
    if (plat->devices[dev_index]) {
        CONFLUX_INFO("[PLATFORM] Closing device %d\n", dev_index);
        conflux_device_destroy(plat->devices[dev_index]);
        plat->devices[dev_index] = NULL;
    }
    
    pthread_mutex_unlock(&plat->lock);
}

/* ---- 关闭全部 ---- */
void conflux_platform_close_all(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (!plat->initialized) return;
    
    CONFLUX_INFO("[PLATFORM] Closing all devices...\n");
    
    for (int i = 0; i < plat->num_devices; i++) {
        conflux_platform_close_device(i);
    }
}

/* ---- 查询 ---- */
int conflux_platform_get_num_devices(void) 
{
    return g_platform.num_devices;
}

conflux_device_t *conflux_platform_get_device(int dev_index) 
{
    if (dev_index < 0 || dev_index >= g_platform.num_devices) {
        return NULL;
    }
    if (g_platform.devices[dev_index] == NULL) {
        CONFLUX_WARN("[PLATFORM] Device %d not opened. Did you forget to check conflux_platform_open_device?\n",
                    dev_index);
    }
    return g_platform.devices[dev_index];
}

conflux_device_t *conflux_platform_get_default_device(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (plat->default_device < 0 ||
        plat->default_device >= plat->num_devices) {
        /* 自动选第一个 */
        if (plat->num_devices > 0) {
            plat->default_device = 0;
            return plat->devices[0];
        }
        return NULL;
    }
    
    return plat->devices[plat->default_device];
}

const conflux_device_desc_t *conflux_platform_get_desc(int dev_index) 
{
    if (dev_index < 0 || dev_index >= g_platform.num_devices) {
        return NULL;
    }
    return &g_platform.descs[dev_index];
}

/* ---- 选设备（简单轮询） ---- */
int conflux_platform_pick_device(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    if (plat->num_devices == 0) return -1;
    if (plat->num_devices == 1) return 0;
    
    /* 选未打开的 */
    for (int i = 0; i < plat->num_devices; i++) {
        if (plat->descs[i].available && plat->devices[i] == NULL) {
            return i;
        }
    }
    
    /* 都打开了，返回默认 */
    return plat->default_device;
}

/* ---- 调试 dump，所有 printf 保留 ---- */
void conflux_platform_dump(void) 
{
    conflux_platform_t *plat = &g_platform;
    
    printf("\n=== Platform State ===\n");
    printf("  initialized:  %d\n", plat->initialized);
    printf("  num_devices:  %d\n", plat->num_devices);
    printf("  default_idx:  %d\n", plat->default_device);
    
    if (plat->num_devices == 0) {
        printf("  (no devices)\n");
        return;
    }
    
    printf("\n  Device list:\n");
    for (int i = 0; i < plat->num_devices; i++) {
        const conflux_device_desc_t *desc = &plat->descs[i];
        printf("  [%d] %-20s vendor=0x%04X device=0x%04X  "
               "mmio=0x%lx size=%luMB  open=%s  available=%s\n",
               i, desc->path,
               desc->vendor_id, desc->device_id,
               (unsigned long)desc->mmio_base,
               (unsigned long)(desc->mmio_size / (1024 * 1024)),
               plat->devices[i] ? "yes" : "no",
               desc->available ? "yes" : "no");
    }
    
    /* 打开的设备详细信息 */
    for (int i = 0; i < plat->num_devices; i++) {
        if (plat->devices[i]) {
            char info[1024];
            conflux_device_query_info(plat->devices[i], info, sizeof(info));
            printf("\n  Device %d details:\n%s", i, info);
        }
    }
}