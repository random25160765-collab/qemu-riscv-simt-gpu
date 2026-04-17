/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GPGPU IOCTL 定义 - 用户空间和内核共享
 */

#ifndef _GPGPU_IOCTL_H
#define _GPGPU_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

/**
 * struct gpgpu_kernel_params - 内核启动参数（组合网格和块维度）
 */
struct gpgpu_kernel_params {
    __u32 grid_dim[3];   /* Grid 维度 (x, y, z) */
    __u32 block_dim[3];  /* Block 维度 (x, y, z) */
    __u64 kernel_addr;   /* 内核代码在 VRAM 中的地址 */
    __u64 args_addr;     /* 参数在 VRAM 中的地址 */
    __u32 shared_mem;    /* 共享内存大小 */
};

/**
 * struct gpgpu_dma_params - DMA 传输参数
 */
struct gpgpu_dma_params {
    __u64 src_addr;      /* 源地址（主机物理地址或 VRAM 偏移） */
    __u64 dst_addr;      /* 目标地址 */
    __u32 size;          /* 传输大小（字节） */
    __u32 flags;         /* 标志位：方向、中断使能等 */
};

#define GPGPU_IOC_MAGIC 'G'

/* 设置网格维度 (输入: __u32[3]) */
#define GPGPU_IOCTL_SET_GRID_DIM    _IOW(GPGPU_IOC_MAGIC, 1, __u32[3])

/* 设置线程块维度 (输入: __u32[3]) */
#define GPGPU_IOCTL_SET_BLOCK_DIM   _IOW(GPGPU_IOC_MAGIC, 2, __u32[3])

/* 启动内核 (无参数) */
#define GPGPU_IOCTL_LAUNCH_KERNEL   _IO(GPGPU_IOC_MAGIC, 3)

/* 等待内核完成，返回状态 (输出: __u32) */
#define GPGPU_IOCTL_WAIT_KERNEL     _IOR(GPGPU_IOC_MAGIC, 4, __u32)

/* 获取设备状态 (输出: __u32) */
#define GPGPU_IOCTL_GET_STATUS      _IOR(GPGPU_IOC_MAGIC, 5, __u32)

/* 获取错误状态 (输出: __u32) */
#define GPGPU_IOCTL_GET_ERROR       _IOR(GPGPU_IOC_MAGIC, 6, __u32)

/* 复位设备 (无参数) */
#define GPGPU_IOCTL_RESET           _IO(GPGPU_IOC_MAGIC, 7)

/* 一次性设置并启动内核 (输入: struct gpgpu_kernel_params) */
#define GPGPU_IOCTL_LAUNCH_PARAMS   _IOW(GPGPU_IOC_MAGIC, 8, struct gpgpu_kernel_params)

/* DMA 传输操作 (输入/输出: struct gpgpu_dma_params) */
#define GPGPU_IOCTL_DMA_XFER             _IOWR(GPGPU_IOC_MAGIC, 9, struct gpgpu_dma_params)

#endif /* _GPGPU_IOCTL_H */