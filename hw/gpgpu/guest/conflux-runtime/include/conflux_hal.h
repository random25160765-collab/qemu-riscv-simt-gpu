#ifndef CONFLUX_HAL_H
#define CONFLUX_HAL_H

#include "conflux_error.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ================================================================
 * 寄存器布局 — 与你的 QEMU 设备 / Linux 驱动保持一致
 * ================================================================ */

/* MMIO 寄存器偏移量（与 GPGPU_REG_* 一一对应） */
#define CONFLUX_REG_DEV_ID           0x0000
#define CONFLUX_REG_DEV_VERSION      0x0004
#define CONFLUX_REG_DEV_CAPS         0x0008
#define CONFLUX_REG_VRAM_SIZE_LO     0x000C
#define CONFLUX_REG_VRAM_SIZE_HI     0x0010
#define CONFLUX_REG_GLOBAL_CTRL      0x0100
#define CONFLUX_REG_GLOBAL_STATUS    0x0104
#define CONFLUX_REG_ERROR_STATUS     0x0108
#define CONFLUX_REG_IRQ_ENABLE       0x0200
#define CONFLUX_REG_IRQ_STATUS       0x0204
#define CONFLUX_REG_IRQ_ACK          0x0208
#define CONFLUX_REG_KERNEL_ADDR_LO   0x0300
#define CONFLUX_REG_KERNEL_ADDR_HI   0x0304
#define CONFLUX_REG_KERNEL_ARGS_LO   0x0308
#define CONFLUX_REG_KERNEL_ARGS_HI   0x030C
#define CONFLUX_REG_GRID_DIM_X       0x0310
#define CONFLUX_REG_GRID_DIM_Y       0x0314
#define CONFLUX_REG_GRID_DIM_Z       0x0318
#define CONFLUX_REG_BLOCK_DIM_X      0x031C
#define CONFLUX_REG_BLOCK_DIM_Y      0x0320
#define CONFLUX_REG_BLOCK_DIM_Z      0x0324
#define CONFLUX_REG_SHARED_MEM_SIZE  0x0328
#define CONFLUX_REG_DISPATCH         0x0330
#define CONFLUX_REG_DMA_SRC_LO       0x0400
#define CONFLUX_REG_DMA_SRC_HI       0x0404
#define CONFLUX_REG_DMA_DST_LO       0x0408
#define CONFLUX_REG_DMA_DST_HI       0x040C
#define CONFLUX_REG_DMA_SIZE         0x0410
#define CONFLUX_REG_DMA_CTRL         0x0414
#define CONFLUX_REG_DMA_STATUS       0x0418
#define CONFLUX_REG_LOG_LEVEL        0x0500
#define CONFLUX_REG_BACKEND_SELECT   0x0600

/* 控制寄存器位域 */
#define CONFLUX_CTRL_ENABLE          (1 << 0)
#define CONFLUX_CTRL_RESET           (1 << 1)

/* 状态寄存器位域 */
#define CONFLUX_STATUS_READY         (1 << 0)
#define CONFLUX_STATUS_BUSY          (1 << 1)
#define CONFLUX_STATUS_ERROR         (1 << 2)

/* 错误寄存器位域 */
#define CONFLUX_ERR_INVALID_CMD      (1 << 0)
#define CONFLUX_ERR_VRAM_FAULT       (1 << 1)
#define CONFLUX_ERR_KERNEL_FAULT     (1 << 2)
#define CONFLUX_ERR_DMA_FAULT        (1 << 3)

/* 中断位 */
#define CONFLUX_IRQ_KERNEL_DONE      (1 << 0)
#define CONFLUX_IRQ_DMA_DONE         (1 << 1)
#define CONFLUX_IRQ_ERROR            (1 << 2)

/* DMA 控制位 */
#define CONFLUX_DMA_START            (1 << 0)
#define CONFLUX_DMA_DIR_FROM_VRAM    (1 << 1)  /* 0=to VRAM, 1=from VRAM */
#define CONFLUX_DMA_IRQ_ENABLE       (1 << 2)

/* DMA 状态位 */
#define CONFLUX_DMA_BUSY             (1 << 0)
#define CONFLUX_DMA_COMPLETE         (1 << 1)
#define CONFLUX_DMA_ERROR            (1 << 2)

/* 后端选择 */
#define CONFLUX_BACKEND_BUILTIN      0
#define CONFLUX_BACKEND_SIMX         1

/* ================================================================
 * HAL 操作模式
 * ================================================================ */

typedef enum {
    CONFLUX_HAL_MODE_MMIO,     /* 直接 MMIO（无驱动 / UIO） */
    CONFLUX_HAL_MODE_IOCTL,    /* 通过 Linux 驱动 ioctl */
    CONFLUX_HAL_MODE_SIM,      /* 纯软件模拟（无硬件） */
} conflux_hal_mode_t;

/* ================================================================
 * HAL 上下文
 * ================================================================ */

typedef struct {
    conflux_hal_mode_t mode;
    
    /* MMIO 模式 */
    int fd;                       /* /dev/mem 或 /dev/uio0 */
    volatile uint32_t *regs;      /* mmap 的寄存器基址 */
    volatile uint8_t  *vram;      /* mmap 的显存基址 */
    size_t vram_size;             /* 显存大小 */
    
    /* IOCTL 模式 */
    int dev_fd;                   /* /dev/gpgpu0（驱动模块名保留 gpgpu） */
    
    /* 通用状态 */
    int last_error;
    bool initialized;
    
} conflux_hal_t;

/* ================================================================
 * API — 模式无关的硬件操作
 * ================================================================ */

/* 初始化/清理 */
int  conflux_hal_init(conflux_hal_t *hal, conflux_hal_mode_t mode,
                     const char *device_path);
void conflux_hal_close(conflux_hal_t *hal);

/* 寄存器读写（自动选择 MMIO 或 IOCTL） */
uint32_t conflux_hal_read_reg(conflux_hal_t *hal, uint32_t offset);
void     conflux_hal_write_reg(conflux_hal_t *hal, uint32_t offset, uint32_t value);

/* 设备内存访问（host <-> device） */
int  conflux_hal_mem_write(conflux_hal_t *hal, uint64_t offset,
                           const void *buf, size_t size);
int  conflux_hal_mem_read(conflux_hal_t *hal, uint64_t offset,
                          void *buf, size_t size);

/* 高层设备命令 */
int  conflux_hal_enable(conflux_hal_t *hal);
int  conflux_hal_reset(conflux_hal_t *hal);
int  conflux_hal_get_status(conflux_hal_t *hal, uint32_t *status);
int  conflux_hal_get_error(conflux_hal_t *hal, uint32_t *error);
int  conflux_hal_wait_ready(conflux_hal_t *hal, uint32_t timeout_ms);

/* 内核启动 */
int  conflux_hal_set_grid_dim(conflux_hal_t *hal, 
                             uint32_t x, uint32_t y, uint32_t z);
int  conflux_hal_set_block_dim(conflux_hal_t *hal,
                               uint32_t x, uint32_t y, uint32_t z);
int  conflux_hal_set_kernel_addr(conflux_hal_t *hal, uint64_t addr);
int  conflux_hal_set_args_addr(conflux_hal_t *hal, uint64_t addr);
int  conflux_hal_set_shared_mem(conflux_hal_t *hal, uint32_t size);
int  conflux_hal_dispatch(conflux_hal_t *hal);
int  conflux_hal_wait_kernel(conflux_hal_t *hal, uint32_t timeout_ms);

/* 组合式启动（一次 ioctl） */
int  conflux_hal_launch_kernel(conflux_hal_t *hal,
                               uint64_t kernel_addr,
                               uint64_t args_addr,
                               const uint32_t grid_dim[3],
                               const uint32_t block_dim[3],
                               uint32_t shared_mem);

/* DMA */
int  conflux_hal_dma_transfer(conflux_hal_t *hal,
                              uint64_t src_addr,
                              uint64_t dst_addr,
                              uint32_t size,
                              bool to_device);

/* 后端控制 */
int  conflux_hal_select_backend(conflux_hal_t *hal, uint32_t backend);

/* 日志控制 */
int  conflux_hal_set_log_level(conflux_hal_t *hal, uint32_t level,
                               uint32_t categories);

/* 查询 */
int  conflux_hal_get_vram_size(conflux_hal_t *hal, uint64_t *size);
int  conflux_hal_get_device_id(conflux_hal_t *hal, uint32_t *id);
int  conflux_hal_get_version(conflux_hal_t *hal, uint32_t *version);

/* 调试 */
void conflux_hal_dump_regs(const conflux_hal_t *hal);

#endif