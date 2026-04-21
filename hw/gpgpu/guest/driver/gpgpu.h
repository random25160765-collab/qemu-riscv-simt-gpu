/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GPGPU Linux Driver - Hardware Definition Header
 * 
 * 此文件与 QEMU hw/misc/gpgpu.h 保持同步
 * 定义了驱动与虚拟硬件之间的寄存器接口
 */

#ifndef _GPGPU_DRIVER_H
#define _GPGPU_DRIVER_H

#include <linux/types.h>
#include <linux/pci.h>

/*
 * ============================================================================
 * PCI 设备标识
 * ============================================================================
 */
#define GPGPU_VENDOR_ID         0x1234
#define GPGPU_DEVICE_ID         0x1337

/*
 * ============================================================================
 * BAR 空间配置
 * ============================================================================
 */
#define GPGPU_BAR_CTRL          0       /* BAR0: 控制寄存器 */
#define GPGPU_BAR_VRAM          2       /* BAR2: 显存 */
#define GPGPU_BAR_DOORBELL      4       /* BAR4: 门铃寄存器 */

/*
 * ============================================================================
 * 控制寄存器偏移量 (BAR0)
 * ============================================================================
 */

/* 设备信息寄存器组 (0x0000 - 0x00FF) - 只读 */
#define GPGPU_REG_DEV_ID            0x0000  /* 设备标识符 "GPPU" */
#define GPGPU_REG_DEV_VERSION       0x0004  /* 版本号 */
#define GPGPU_REG_DEV_CAPS          0x0008  /* 设备能力 */
#define GPGPU_REG_VRAM_SIZE_LO      0x000C  /* 显存大小低 32 位 */
#define GPGPU_REG_VRAM_SIZE_HI      0x0010  /* 显存大小高 32 位 */

/* 全局控制寄存器组 (0x0100 - 0x01FF) */
#define GPGPU_REG_GLOBAL_CTRL       0x0100  /* 全局控制 */
#define GPGPU_REG_GLOBAL_STATUS     0x0104  /* 全局状态 */
#define GPGPU_REG_ERROR_STATUS      0x0108  /* 错误状态 (写 1 清除) */

/* 中断控制寄存器组 (0x0200 - 0x02FF) */
#define GPGPU_REG_IRQ_ENABLE        0x0200  /* 中断使能掩码 */
#define GPGPU_REG_IRQ_STATUS        0x0204  /* 中断状态 */
#define GPGPU_REG_IRQ_ACK           0x0208  /* 中断确认 (写 1 清除) */

/* 内核分发寄存器组 (0x0300 - 0x03FF) */
#define GPGPU_REG_KERNEL_ADDR_LO    0x0300  /* 内核代码地址低 32 位 */
#define GPGPU_REG_KERNEL_ADDR_HI    0x0304  /* 内核代码地址高 32 位 */
#define GPGPU_REG_KERNEL_ARGS_LO    0x0308  /* 参数地址低 32 位 */
#define GPGPU_REG_KERNEL_ARGS_HI    0x030C  /* 参数地址高 32 位 */
#define GPGPU_REG_GRID_DIM_X        0x0310  /* Grid X 维度 */
#define GPGPU_REG_GRID_DIM_Y        0x0314  /* Grid Y 维度 */
#define GPGPU_REG_GRID_DIM_Z        0x0318  /* Grid Z 维度 */
#define GPGPU_REG_BLOCK_DIM_X       0x031C  /* Block X 维度 */
#define GPGPU_REG_BLOCK_DIM_Y       0x0320  /* Block Y 维度 */
#define GPGPU_REG_BLOCK_DIM_Z       0x0324  /* Block Z 维度 */
#define GPGPU_REG_SHARED_MEM_SIZE   0x0328  /* 共享内存大小 */
#define GPGPU_REG_DISPATCH          0x0330  /* 启动内核 (写任意值) */

/* 日志控制寄存器组 (0x0500) */
#define GPGPU_REG_LOG_LEVEL         0x0500  /* 日志级别控制
                                             *   bits[7:0]  : 级别 (0=OFF..6=TRACE)
                                             *   bits[15:8] : 类别掩码 (0=不修改) */

/* DMA 引擎寄存器组 (0x0400 - 0x04FF) */
#define GPGPU_REG_DMA_SRC_LO        0x0400  /* 源地址低 32 位 */
#define GPGPU_REG_DMA_SRC_HI        0x0404  /* 源地址高 32 位 */
#define GPGPU_REG_DMA_DST_LO        0x0408  /* 目标地址低 32 位 */
#define GPGPU_REG_DMA_DST_HI        0x040C  /* 目标地址高 32 位 */
#define GPGPU_REG_DMA_SIZE          0x0410  /* 传输大小 (字节) */
#define GPGPU_REG_DMA_CTRL          0x0414  /* DMA 控制 */
#define GPGPU_REG_DMA_STATUS        0x0418  /* DMA 状态 */

/* 线程上下文寄存器组 (0x1000 - 0x1FFF) - 只读 */
#define GPGPU_REG_THREAD_ID_X       0x1000  /* threadIdx.x */
#define GPGPU_REG_THREAD_ID_Y       0x1004  /* threadIdx.y */
#define GPGPU_REG_THREAD_ID_Z       0x1008  /* threadIdx.z */
#define GPGPU_REG_BLOCK_ID_X        0x1010  /* blockIdx.x */
#define GPGPU_REG_BLOCK_ID_Y        0x1014  /* blockIdx.y */
#define GPGPU_REG_BLOCK_ID_Z        0x1018  /* blockIdx.z */
#define GPGPU_REG_WARP_ID           0x1020  /* warp ID */
#define GPGPU_REG_LANE_ID           0x1024  /* lane ID (0-31) */

/* 同步寄存器组 (0x2000 - 0x2FFF) */
#define GPGPU_REG_BARRIER           0x2000  /* barrier 触发 (写任意值) */
#define GPGPU_REG_THREAD_MASK       0x2004  /* 活跃线程掩码 */

/*
 * ============================================================================
 * 寄存器位域定义
 * ============================================================================
 */

/* GLOBAL_CTRL 位 */
#define GPGPU_CTRL_ENABLE           BIT(0)  /* 设备使能 */
#define GPGPU_CTRL_RESET            BIT(1)  /* 软复位 */

/* GLOBAL_STATUS 位 */
#define GPGPU_STATUS_READY          BIT(0)  /* 设备就绪 */
#define GPGPU_STATUS_BUSY           BIT(1)  /* 设备忙 */
#define GPGPU_STATUS_ERROR          BIT(2)  /* 发生错误 */

/* ERROR_STATUS 位 */
#define GPGPU_ERR_INVALID_CMD       BIT(0)  /* 无效命令 */
#define GPGPU_ERR_VRAM_FAULT        BIT(1)  /* 显存访问越界 */
#define GPGPU_ERR_KERNEL_FAULT      BIT(2)  /* 内核执行错误 */
#define GPGPU_ERR_DMA_FAULT         BIT(3)  /* DMA 错误 */

/* 中断位 */
#define GPGPU_IRQ_KERNEL_DONE       BIT(0)  /* 内核执行完成 */
#define GPGPU_IRQ_DMA_DONE          BIT(1)  /* DMA 完成 */
#define GPGPU_IRQ_ERROR             BIT(2)  /* 错误中断 */

/* DMA_CTRL 位 */
#define GPGPU_DMA_START             BIT(0)  /* 启动 DMA */
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)
#define GPGPU_DMA_DIR_FROM_VRAM     BIT(1)  /* 方向位 */
#define GPGPU_DMA_IRQ_ENABLE        BIT(2)  /* 完成时产生中断 */

/* DMA_STATUS 位 */
#define GPGPU_DMA_BUSY              BIT(0)  /* DMA 忙 */
#define GPGPU_DMA_COMPLETE          BIT(1)  /* DMA 完成 */
#define GPGPU_DMA_ERROR             BIT(2)  /* DMA 错误 */

/* 设备标识常量 */
#define GPGPU_DEV_ID_VALUE          0x47505055  /* "GPPU" ASCII */

/*
 * ============================================================================
 * 驱动私有数据结构
 * ============================================================================
 */

/**
 * struct gpgpu_dev - GPGPU 设备私有数据结构
 * @pdev:         PCI 设备指针
 * @bar0:         BAR0 映射地址 (控制寄存器)
 * @bar2:         BAR2 映射地址 (VRAM)
 * @bar2_size:    VRAM 大小 (字节)
 * @bar4:         BAR4 映射地址 (门铃寄存器)
 * @dev_id:       设备 ID (用于字符设备)
 * @cdev:         字符设备结构
 * @cdev_created: 字符设备是否已创建
 */
struct gpgpu_dev {
    struct pci_dev *pdev;

    void __iomem *bar0;
    void __iomem *bar2;
    resource_size_t bar2_size;
    void __iomem *bar4;

    /* 字符设备相关 */
    int dev_id;
    struct cdev cdev;
    struct device *device;
    bool cdev_created;

    /* 中断相关 */
    int irq;                            // 中断号
    wait_queue_head_t kernel_wq;        // 等待队列头
    int kernel_completed;               // 内核完成标志
    int error_occurred;                 // 错误发生标志
    bool irq_enabled;                   // 中断是否已启用

    /* DMA 相关 */
    bool dma_in_progress;               // DMA 是否进行中
};

/*
 * ============================================================================
 * 寄存器读写辅助函数
 * ============================================================================
 */

/**
 * gpgpu_writel - 写入 32 位值到控制寄存器
 * @gdev: 设备私有数据
 * @reg:  寄存器偏移量 (字节)
 * @val:  要写入的值
 */
static inline void gpgpu_writel(struct gpgpu_dev *gdev, u32 reg, u32 val)
{
    writel(val, gdev->bar0 + reg);
}

/**
 * gpgpu_readl - 从控制寄存器读取 32 位值
 * @gdev: 设备私有数据
 * @reg:  寄存器偏移量 (字节)
 * @return: 寄存器值
 */
static inline u32 gpgpu_readl(struct gpgpu_dev *gdev, u32 reg)
{
    return readl(gdev->bar0 + reg);
}

/*
 * ============================================================================
 * VRAM 访问辅助函数
 * ============================================================================
 */

/**
 * gpgpu_vram_write - 写入数据到 VRAM
 * @gdev:  设备私有数据
 * @offset: VRAM 内偏移 (字节)
 * @buf:   源数据缓冲区
 * @size:  写入大小 (字节)
 */
static inline void gpgpu_vram_write(struct gpgpu_dev *gdev, u32 offset,
                                    const void *buf, size_t size)
{
    if (offset + size <= gdev->bar2_size)
        memcpy_toio(gdev->bar2 + offset, buf, size);
}

/**
 * gpgpu_vram_read - 从 VRAM 读取数据
 * @gdev:  设备私有数据
 * @offset: VRAM 内偏移 (字节)
 * @buf:   目标缓冲区
 * @size:  读取大小 (字节)
 */
static inline void gpgpu_vram_read(struct gpgpu_dev *gdev, u32 offset,
                                   void *buf, size_t size)
{
    if (offset + size <= gdev->bar2_size)
        memcpy_fromio(buf, gdev->bar2 + offset, size);
}

/*
 * ============================================================================
 * 字符设备接口 (gpgpu_cdev.c)
 * ============================================================================
 */

int gpgpu_cdev_init(struct gpgpu_dev *gdev);
void gpgpu_cdev_cleanup(struct gpgpu_dev *gdev);

#endif /* _GPGPU_DRIVER_H */
