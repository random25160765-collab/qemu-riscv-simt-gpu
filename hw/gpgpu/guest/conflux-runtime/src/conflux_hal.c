#include "conflux_hal.h"
#include "conflux_log.h"     /* 用你自己的日志系统 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <errno.h>

/* 
 * 如果编译时能找到 gpgpu_ioctl.h 就 include，否则用硬编码的 ioctl 定义。
 */
#ifdef HAVE_GPGPU_IOCTL_H
#include "gpgpu_ioctl.h"
#else
/* 硬编码 ioctl 定义 — 与你的驱动头文件完全一致 */
struct gpgpu_kernel_params {
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t kernel_addr;
    uint64_t args_addr;
    uint32_t shared_mem;
};

struct gpgpu_dma_params {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t flags;
};

struct gpgpu_log_params {
    uint32_t level;
    uint32_t categories;
};

struct gpgpu_backend_params {
    uint32_t backend;
};

#define GPGPU_IOC_MAGIC         'G'
#define GPGPU_IOCTL_SET_GRID_DIM    _IOW(GPGPU_IOC_MAGIC, 1, uint32_t[3])
#define GPGPU_IOCTL_SET_BLOCK_DIM   _IOW(GPGPU_IOC_MAGIC, 2, uint32_t[3])
#define GPGPU_IOCTL_LAUNCH_KERNEL   _IO(GPGPU_IOC_MAGIC, 3)
#define GPGPU_IOCTL_WAIT_KERNEL     _IOR(GPGPU_IOC_MAGIC, 4, uint32_t)
#define GPGPU_IOCTL_GET_STATUS      _IOR(GPGPU_IOC_MAGIC, 5, uint32_t)
#define GPGPU_IOCTL_GET_ERROR       _IOR(GPGPU_IOC_MAGIC, 6, uint32_t)
#define GPGPU_IOCTL_RESET           _IO(GPGPU_IOC_MAGIC, 7)
#define GPGPU_IOCTL_LAUNCH_PARAMS   _IOW(GPGPU_IOC_MAGIC, 8, struct gpgpu_kernel_params)
#define GPGPU_IOCTL_DMA_XFER        _IOWR(GPGPU_IOC_MAGIC, 9, struct gpgpu_dma_params)
#define GPGPU_IOCTL_SET_LOG_LEVEL   _IOW(GPGPU_IOC_MAGIC, 10, struct gpgpu_log_params)
#define GPGPU_IOCTL_SET_BACKEND     _IOW(GPGPU_IOC_MAGIC, 11, struct gpgpu_backend_params)

#define GPGPU_DMA_START             (1 << 0)
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)
#define GPGPU_DMA_IRQ_ENABLE        (1 << 2)

#define GPGPU_BACKEND_BUILTIN       0
#define GPGPU_BACKEND_SIMX          1
#endif

/* ---- 初始化 ---- */
int conflux_hal_init(conflux_hal_t *hal, conflux_hal_mode_t mode,
                    const char *device_path) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    memset(hal, 0, sizeof(conflux_hal_t));
    hal->mode = mode;
    
    switch (mode) {
        case CONFLUX_HAL_MODE_IOCTL: {
            if (!device_path) device_path = "/dev/gpgpu0";
            
            hal->dev_fd = open(device_path, O_RDWR);
            if (hal->dev_fd < 0) {
                CONFLUX_ERROR("Failed to open %s: %s", device_path, strerror(errno));
                return CONFLUX_ERR_DEVICE_NOT_READY;
            }
            
            CONFLUX_INFO("HAL: IOCTL mode, fd=%d, path=%s", hal->dev_fd, device_path);
            break;
        }
        
        case CONFLUX_HAL_MODE_MMIO: {
            /* 留空 — 以后做无内核驱动再用 */
            CONFLUX_ERROR("HAL: MMIO mode not yet implemented");
            return CONFLUX_ERR_INVALID;
        }
        
        case CONFLUX_HAL_MODE_SIM: {
            CONFLUX_INFO("HAL: Simulation mode (no hardware)");
            break;
        }
        
        default:
            return CONFLUX_ERR_INVALID;
    }
    
    hal->initialized = true;
    
    /* 使能设备 */
    int ret = conflux_hal_enable(hal);
    if (ret != CONFLUX_SUCCESS) {
        CONFLUX_WARN("HAL: device enable failed (may already be enabled)");
    }
    
    return CONFLUX_SUCCESS;
}

/* ---- 关闭 ---- */
void conflux_hal_close(conflux_hal_t *hal) 
{
    if (!hal || !hal->initialized) return;
    
    CONFLUX_INFO("HAL: Closing");
    
    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL:
            if (hal->dev_fd >= 0) {
                close(hal->dev_fd);
                hal->dev_fd = -1;
            }
            break;
            
        case CONFLUX_HAL_MODE_MMIO:
            if (hal->regs) {
                munmap((void *)hal->regs, 0x10000);
                hal->regs = NULL;
            }
            if (hal->fd >= 0) {
                close(hal->fd);
                hal->fd = -1;
            }
            break;
            
        case CONFLUX_HAL_MODE_SIM:
            break;
    }
    
    hal->initialized = false;
}

/* ---- 寄存器读写（IOCTL 模式下通过 ioctl，MMIO 模式下直接访问） ---- */
uint32_t conflux_hal_read_reg(conflux_hal_t *hal, uint32_t offset) 
{
    if (!hal || !hal->initialized) return 0;
    
    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL: {
            uint32_t val = 0;
            int ret;
            
            switch (offset) {
                case CONFLUX_REG_GLOBAL_STATUS:
                    ret = ioctl(hal->dev_fd, GPGPU_IOCTL_GET_STATUS, &val);
                    if (ret < 0) val = 0;
                    break;
                case CONFLUX_REG_ERROR_STATUS:
                    ret = ioctl(hal->dev_fd, GPGPU_IOCTL_GET_ERROR, &val);
                    if (ret < 0) val = 0;
                    break;
                default:
                    CONFLUX_WARN("HAL: read_reg 0x%04x not supported via IOCTL", offset);
                    val = 0;
                    break;
            }
            return val;
        }
        
        case CONFLUX_HAL_MODE_MMIO:
            if (hal->regs) {
                return *(volatile uint32_t *)((uint8_t *)hal->regs + offset);
            }
            return 0;
            
        case CONFLUX_HAL_MODE_SIM:
            /* 可以返回假状态 */
            if (offset == CONFLUX_REG_GLOBAL_STATUS) {
                return CONFLUX_STATUS_READY;
            }
            return 0;
    }
    
    return 0;
}

void conflux_hal_write_reg(conflux_hal_t *hal, uint32_t offset, uint32_t value) 
{
    if (!hal || !hal->initialized) return;
    
    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL:
            /* 大多数写寄存器通过专门的函数，不经过这里 */
            CONFLUX_DEBUG("HAL: write_reg 0x%04x = 0x%x (IOCTL — use dedicated API)", 
                         offset, value);
            break;
            
        case CONFLUX_HAL_MODE_MMIO:
            if (hal->regs) {
                *(volatile uint32_t *)((uint8_t *)hal->regs + offset) = value;
            }
            break;
            
        case CONFLUX_HAL_MODE_SIM:
            break;
    }
}

/* ---- VRAM 访问 ---- */
/*
 * conflux_hal_mem_write — host buffer 写入设备显存
 *
 * IOCTL 模式契约：把 host 用户态 buf 的虚拟地址作为 src_addr 传给驱动，
 * 驱动负责通过 copy_from_user 拷贝到设备 VRAM。
 */
int conflux_hal_mem_write(conflux_hal_t *hal, uint64_t offset,
                          const void *buf, size_t size)
{
    if (!hal || !buf || size == 0) return CONFLUX_ERR_INVALID;

    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL:
            return conflux_hal_dma_transfer(hal,
                                            (uint64_t)(uintptr_t)buf,
                                            offset,
                                            (uint32_t)size,
                                            true /* to_device */);

        case CONFLUX_HAL_MODE_MMIO:
            if (hal->vram && offset + size <= hal->vram_size) {
                memcpy((void *)(hal->vram + offset), buf, size);
                return CONFLUX_SUCCESS;
            }
            return CONFLUX_ERR_MEM_INVALID_ADDR;

        case CONFLUX_HAL_MODE_SIM:
            CONFLUX_DEBUG("HAL: SIM mem_write 0x%lx, %zu bytes (ignored)",
                         (unsigned long)offset, size);
            return CONFLUX_SUCCESS;
    }

    return CONFLUX_ERR_INVALID;
}

int conflux_hal_mem_read(conflux_hal_t *hal, uint64_t offset,
                         void *buf, size_t size)
{
    if (!hal || !buf || size == 0) return CONFLUX_ERR_INVALID;

    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL:
            return conflux_hal_dma_transfer(hal,
                                            offset,
                                            (uint64_t)(uintptr_t)buf,
                                            (uint32_t)size,
                                            false /* from device */);

        case CONFLUX_HAL_MODE_MMIO:
            if (hal->vram && offset + size <= hal->vram_size) {
                memcpy(buf, (const void *)(hal->vram + offset), size);
                return CONFLUX_SUCCESS;
            }
            return CONFLUX_ERR_MEM_INVALID_ADDR;

        case CONFLUX_HAL_MODE_SIM:
            memset(buf, 0, size);
            return CONFLUX_SUCCESS;
    }

    return CONFLUX_ERR_INVALID;
}

/* ---- 设备控制 ---- */
int conflux_hal_enable(conflux_hal_t *hal) 
{
    if (!hal || !hal->initialized) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_RESET);
        if (ret < 0) {
            CONFLUX_WARN("HAL: reset failed: %s", strerror(errno));
        }
    }
    
    return conflux_hal_wait_ready(hal, 1000);
}

int conflux_hal_reset(conflux_hal_t *hal) 
{
    if (!hal || !hal->initialized) return CONFLUX_ERR_INVALID;
    
    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL: {
            int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_RESET);
            if (ret < 0) {
                CONFLUX_ERROR("HAL: reset ioctl failed: %s", strerror(errno));
                return CONFLUX_ERR_DEVICE_RESET_FAIL;
            }
            return CONFLUX_SUCCESS;
        }
        case CONFLUX_HAL_MODE_MMIO:
            conflux_hal_write_reg(hal, CONFLUX_REG_GLOBAL_CTRL, CONFLUX_CTRL_RESET);
            usleep(10000);
            return conflux_hal_wait_ready(hal, 1000);
        case CONFLUX_HAL_MODE_SIM:
            return CONFLUX_SUCCESS;
    }
    
    return CONFLUX_ERR_INVALID;
}

int conflux_hal_get_status(conflux_hal_t *hal, uint32_t *status) 
{
    if (!hal || !status) return CONFLUX_ERR_INVALID;
    *status = conflux_hal_read_reg(hal, CONFLUX_REG_GLOBAL_STATUS);
    return CONFLUX_SUCCESS;
}

int conflux_hal_get_error(conflux_hal_t *hal, uint32_t *error) 
{
    if (!hal || !error) return CONFLUX_ERR_INVALID;
    *error = conflux_hal_read_reg(hal, CONFLUX_REG_ERROR_STATUS);
    return CONFLUX_SUCCESS;
}

int conflux_hal_wait_ready(conflux_hal_t *hal, uint32_t timeout_ms) 
{
    uint32_t status;
    for (uint32_t i = 0; i < timeout_ms; i++) {
        conflux_hal_get_status(hal, &status);
        if (status & CONFLUX_STATUS_READY) return CONFLUX_SUCCESS;
        usleep(1000);
    }
    return CONFLUX_ERR_TIMEOUT;
}

/* ---- 内核启动 ---- */
int conflux_hal_set_grid_dim(conflux_hal_t *hal, 
                            uint32_t x, uint32_t y, uint32_t z) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        uint32_t dim[3] = {x, y, z};
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_SET_GRID_DIM, dim);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: SET_GRID_DIM failed: %s", strerror(errno));
            return CONFLUX_ERR_INVALID;
        }
        return CONFLUX_SUCCESS;
    }
    
    /* MMIO 模式 */
    conflux_hal_write_reg(hal, CONFLUX_REG_GRID_DIM_X, x);
    conflux_hal_write_reg(hal, CONFLUX_REG_GRID_DIM_Y, y);
    conflux_hal_write_reg(hal, CONFLUX_REG_GRID_DIM_Z, z);
    return CONFLUX_SUCCESS;
}

int conflux_hal_set_block_dim(conflux_hal_t *hal,
                              uint32_t x, uint32_t y, uint32_t z) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        uint32_t dim[3] = {x, y, z};
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_SET_BLOCK_DIM, dim);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: SET_BLOCK_DIM failed: %s", strerror(errno));
            return CONFLUX_ERR_INVALID;
        }
        return CONFLUX_SUCCESS;
    }
    
    conflux_hal_write_reg(hal, CONFLUX_REG_BLOCK_DIM_X, x);
    conflux_hal_write_reg(hal, CONFLUX_REG_BLOCK_DIM_Y, y);
    conflux_hal_write_reg(hal, CONFLUX_REG_BLOCK_DIM_Z, z);
    return CONFLUX_SUCCESS;
}

int conflux_hal_set_kernel_addr(conflux_hal_t *hal, uint64_t addr) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    conflux_hal_write_reg(hal, CONFLUX_REG_KERNEL_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    conflux_hal_write_reg(hal, CONFLUX_REG_KERNEL_ADDR_HI, (uint32_t)(addr >> 32));
    return CONFLUX_SUCCESS;
}

int conflux_hal_set_args_addr(conflux_hal_t *hal, uint64_t addr) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    conflux_hal_write_reg(hal, CONFLUX_REG_KERNEL_ARGS_LO, (uint32_t)(addr & 0xFFFFFFFF));
    conflux_hal_write_reg(hal, CONFLUX_REG_KERNEL_ARGS_HI, (uint32_t)(addr >> 32));
    return CONFLUX_SUCCESS;
}

int conflux_hal_set_shared_mem(conflux_hal_t *hal, uint32_t size) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    conflux_hal_write_reg(hal, CONFLUX_REG_SHARED_MEM_SIZE, size);
    return CONFLUX_SUCCESS;
}

int conflux_hal_dispatch(conflux_hal_t *hal) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_LAUNCH_KERNEL);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: LAUNCH_KERNEL failed: %s", strerror(errno));
            return CONFLUX_ERR_DEVICE_FAULT;
        }
        return CONFLUX_SUCCESS;
    }
    
    conflux_hal_write_reg(hal, CONFLUX_REG_DISPATCH, 1);
    return CONFLUX_SUCCESS;
}

int conflux_hal_wait_kernel(conflux_hal_t *hal, uint32_t timeout_ms)
{
    if (!hal) return CONFLUX_ERR_INVALID;

    /* timeout_ms == 0 解读为"无限等" */
    uint32_t budget = (timeout_ms == 0) ? UINT32_MAX : timeout_ms;

    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL: {
            uint32_t status = 0;
            int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_WAIT_KERNEL, &status);
            if (ret < 0) {
                CONFLUX_ERROR("HAL: WAIT_KERNEL failed: %s", strerror(errno));
                return CONFLUX_ERR_TIMEOUT;
            }
            return CONFLUX_SUCCESS;
        }

        case CONFLUX_HAL_MODE_SIM:
            /* SIM 设备永远 idle */
            return CONFLUX_SUCCESS;

        case CONFLUX_HAL_MODE_MMIO: {
            uint32_t status;
            for (uint32_t i = 0; i < budget; i++) {
                conflux_hal_get_status(hal, &status);
                if (!(status & CONFLUX_STATUS_BUSY)) return CONFLUX_SUCCESS;
                usleep(1000);
            }
            return CONFLUX_ERR_TIMEOUT;
        }
    }
    return CONFLUX_ERR_INVALID;
}

/* ---- 组合式内核启动（一次 ioctl） ---- */
int conflux_hal_launch_kernel(conflux_hal_t *hal,
                              uint64_t kernel_addr,
                              uint64_t args_addr,
                              const uint32_t grid_dim[3],
                              const uint32_t block_dim[3],
                              uint32_t shared_mem) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        struct gpgpu_kernel_params params = {
            .kernel_addr = kernel_addr,
            .args_addr   = args_addr,
            .shared_mem  = shared_mem,
        };
        memcpy(params.grid_dim, grid_dim, sizeof(params.grid_dim));
        memcpy(params.block_dim, block_dim, sizeof(params.block_dim));
        
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: LAUNCH_PARAMS failed: %s", strerror(errno));
            return CONFLUX_ERR_DEVICE_FAULT;
        }
        return CONFLUX_SUCCESS;
    }
    
    /* MMIO 模式：分步写寄存器 */
    int ret;
    ret = conflux_hal_set_kernel_addr(hal, kernel_addr); if (ret) return ret;
    ret = conflux_hal_set_args_addr(hal, args_addr);     if (ret) return ret;
    ret = conflux_hal_set_grid_dim(hal, grid_dim[0], grid_dim[1], grid_dim[2]); if (ret) return ret;
    ret = conflux_hal_set_block_dim(hal, block_dim[0], block_dim[1], block_dim[2]); if (ret) return ret;
    ret = conflux_hal_set_shared_mem(hal, shared_mem);    if (ret) return ret;
    
    return conflux_hal_dispatch(hal);
}

/* ---- DMA ---- */
int conflux_hal_dma_transfer(conflux_hal_t *hal,
                             uint64_t src_addr,
                             uint64_t dst_addr,
                             uint32_t size,
                             bool to_device) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    /* CONFLUX_DMA_* 与驱动侧 GPGPU_DMA_* 数值一致（位 0/1/2） */
    struct gpgpu_dma_params dma = {
        .src_addr = src_addr,
        .dst_addr = dst_addr,
        .size     = size,
        .flags    = CONFLUX_DMA_START | CONFLUX_DMA_IRQ_ENABLE,
    };

    if (!to_device) {
        dma.flags |= CONFLUX_DMA_DIR_FROM_VRAM;
    }
    
    switch (hal->mode) {
        case CONFLUX_HAL_MODE_IOCTL: {
            int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_DMA_XFER, &dma);
            if (ret < 0) {
                CONFLUX_ERROR("HAL: DMA_XFER failed: %s", strerror(errno));
                return CONFLUX_ERR_DEVICE_FAULT;
            }
            return CONFLUX_SUCCESS;
        }

        case CONFLUX_HAL_MODE_SIM:
            CONFLUX_DEBUG("HAL: SIM dma_transfer src=0x%lx dst=0x%lx size=%u to_dev=%d (no-op)",
                         (unsigned long)src_addr, (unsigned long)dst_addr,
                         size, (int)to_device);
            return CONFLUX_SUCCESS;

        case CONFLUX_HAL_MODE_MMIO:
            CONFLUX_ERROR("HAL: dma_transfer not supported in MMIO mode yet");
            return CONFLUX_ERR_INVALID;
    }

    return CONFLUX_ERR_INVALID;
}

/* ---- 后端 ---- */
int conflux_hal_select_backend(conflux_hal_t *hal, uint32_t backend) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        struct gpgpu_backend_params params = { .backend = backend };
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_SET_BACKEND, &params);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: SET_BACKEND to %u failed: %s", backend, strerror(errno));
            return CONFLUX_ERR_INVALID;
        }
        CONFLUX_INFO("HAL: Backend switched to %s", 
                    backend == CONFLUX_BACKEND_SIMX ? "SimX" : "Built-in");
        return CONFLUX_SUCCESS;
    }
    
    conflux_hal_write_reg(hal, CONFLUX_REG_BACKEND_SELECT, backend);
    return CONFLUX_SUCCESS;
}

/* ---- 日志 ---- */
int conflux_hal_set_log_level(conflux_hal_t *hal, uint32_t level,
                              uint32_t categories) 
{
    if (!hal) return CONFLUX_ERR_INVALID;
    
    if (hal->mode == CONFLUX_HAL_MODE_IOCTL) {
        struct gpgpu_log_params params = {
            .level      = level,
            .categories = categories,
        };
        int ret = ioctl(hal->dev_fd, GPGPU_IOCTL_SET_LOG_LEVEL, &params);
        if (ret < 0) {
            CONFLUX_ERROR("HAL: SET_LOG_LEVEL failed: %s", strerror(errno));
            return CONFLUX_ERR_INVALID;
        }
        return CONFLUX_SUCCESS;
    }
    
    conflux_hal_write_reg(hal, CONFLUX_REG_LOG_LEVEL, (categories << 8) | level);
    return CONFLUX_SUCCESS;
}

/* ---- 查询 ---- */
int conflux_hal_get_vram_size(conflux_hal_t *hal, uint64_t *size) 
{
    if (!hal || !size) return CONFLUX_ERR_INVALID;
    
    uint32_t lo = conflux_hal_read_reg(hal, CONFLUX_REG_VRAM_SIZE_LO);
    uint32_t hi = conflux_hal_read_reg(hal, CONFLUX_REG_VRAM_SIZE_HI);
    
    *size = ((uint64_t)hi << 32) | lo;
    return CONFLUX_SUCCESS;
}

int conflux_hal_get_device_id(conflux_hal_t *hal, uint32_t *id) 
{
    if (!hal || !id) return CONFLUX_ERR_INVALID;
    *id = conflux_hal_read_reg(hal, CONFLUX_REG_DEV_ID);
    return CONFLUX_SUCCESS;
}

int conflux_hal_get_version(conflux_hal_t *hal, uint32_t *version) 
{
    if (!hal || !version) return CONFLUX_ERR_INVALID;
    *version = conflux_hal_read_reg(hal, CONFLUX_REG_DEV_VERSION);
    return CONFLUX_SUCCESS;
}

/* ---- 调试 ---- */
void conflux_hal_dump_regs(const conflux_hal_t *hal) 
{
    if (!hal || !hal->initialized) {
        printf("HAL: not initialized\n");
        return;
    }
    
    printf("\n=== HAL Register Dump (mode=%d) ===\n", hal->mode);
    
    uint32_t dev_id, version, status, error;
    uint64_t vram_size;
    
    conflux_hal_get_device_id((conflux_hal_t *)hal, &dev_id);
    conflux_hal_get_version((conflux_hal_t *)hal, &version);
    conflux_hal_get_status((conflux_hal_t *)hal, &status);
    conflux_hal_get_error((conflux_hal_t *)hal, &error);
    conflux_hal_get_vram_size((conflux_hal_t *)hal, &vram_size);
    
    printf("  DEV_ID:       0x%08X\n", dev_id);
    printf("  VERSION:      %d\n", version);
    printf("  VRAM_SIZE:    %lu MB\n", (unsigned long)(vram_size / (1024*1024)));
    printf("  STATUS:       0x%X (ready=%d, busy=%d, error=%d)\n",
           status,
           !!(status & CONFLUX_STATUS_READY),
           !!(status & CONFLUX_STATUS_BUSY),
           !!(status & CONFLUX_STATUS_ERROR));
    printf("  ERROR:        0x%X\n", error);
}