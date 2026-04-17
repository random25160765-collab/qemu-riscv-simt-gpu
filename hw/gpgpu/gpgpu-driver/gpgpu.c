// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/bitmap.h>
#include <linux/delay.h> 
#include "gpgpu.h"
#include "gpgpu_ioctl.h"

#define GPGPU_MAX_DEVICES 16

static dev_t gpgpu_devt;
static struct class *gpgpu_class;
static DECLARE_BITMAP(gpgpu_devices_bitmap, GPGPU_MAX_DEVICES);  // 替换 IDA
static DEFINE_SPINLOCK(gpgpu_devices_lock);  // 保护位图操作

/* ========== 设备 ID 管理 ========== */

static int gpgpu_alloc_dev_id(void)
{
    int id;
    unsigned long flags;
    
    spin_lock_irqsave(&gpgpu_devices_lock, flags);
    id = find_first_zero_bit(gpgpu_devices_bitmap, GPGPU_MAX_DEVICES);
    if (id < GPGPU_MAX_DEVICES)
        __set_bit(id, gpgpu_devices_bitmap);
    spin_unlock_irqrestore(&gpgpu_devices_lock, flags);
    
    return (id < GPGPU_MAX_DEVICES) ? id : -ENOSPC;
}

static void gpgpu_free_dev_id(int id)
{
    unsigned long flags;
    
    if (id >= 0 && id < GPGPU_MAX_DEVICES) {
        spin_lock_irqsave(&gpgpu_devices_lock, flags);
        __clear_bit(id, gpgpu_devices_bitmap);
        spin_unlock_irqrestore(&gpgpu_devices_lock, flags);
    }
}

/* ========== 文件操作 ========== */

static int gpgpu_open(struct inode *inode, struct file *filp)
{
    struct gpgpu_dev *gdev = container_of(inode->i_cdev, struct gpgpu_dev, cdev);
    
    if (!gdev || !gdev->pdev) {
        pr_err("gpgpu: Invalid device in open\n");
        return -ENODEV;
    }
    
    filp->private_data = gdev;
    dev_info(&gdev->pdev->dev, "Device opened\n");
    return 0;
}

static int gpgpu_release(struct inode *inode, struct file *filp)
{
    struct gpgpu_dev *gdev = filp->private_data;
    
    if (gdev && gdev->pdev)
        dev_info(&gdev->pdev->dev, "Device closed\n");
    
    return 0;
}

static int gpgpu_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct gpgpu_dev *gdev = filp->private_data;
    resource_size_t paddr;
    unsigned long pfn;
    size_t size;
    int ret;

    if (!gdev || !gdev->pdev)
        return -ENODEV;

    paddr = pci_resource_start(gdev->pdev, GPGPU_BAR_VRAM);
    size = vma->vm_end - vma->vm_start;

    // 边界检查
    if (size > gdev->bar2_size) {
        dev_err(&gdev->pdev->dev, "mmap size %zu exceeds VRAM size %llu\n",
                size, (u64)gdev->bar2_size);
        return -EINVAL;
    }

    // 只允许从VRAM起始位置映射
    if (vma->vm_pgoff != 0) {
        dev_err(&gdev->pdev->dev, "mmap offset must be 0\n");
        return -EINVAL;
    }

    pfn = paddr >> PAGE_SHIFT;
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    ret = remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
    if (ret) {
        dev_err(&gdev->pdev->dev, "remap_pfn_range failed: %d\n", ret);
        return -EAGAIN;
    }

    dev_dbg(&gdev->pdev->dev, "VRAM mmap: vaddr=0x%lx, size=%zu, paddr=0x%llx\n",
            vma->vm_start, size, (u64)paddr);

    return 0;
}

/* ========== 中断处理 ========== */

static irqreturn_t gpgpu_irq_handler(int irq, void *data)
{
    struct gpgpu_dev *gdev = data;
    u32 irq_status;

    if (!gdev || !gdev->bar0)
        return IRQ_NONE;

    // 读取中断状态寄存器
    irq_status = gpgpu_readl(gdev, GPGPU_REG_IRQ_STATUS);

    dev_dbg(&gdev->pdev->dev, "IRQ received, status=0x%08x\n", irq_status);

    // 检查是否是我们的中断
    if (!(irq_status & (GPGPU_IRQ_KERNEL_DONE | GPGPU_IRQ_ERROR | GPGPU_IRQ_DMA_DONE)))
        return IRQ_NONE;

    // 处理内核完成中断
    if (irq_status & GPGPU_IRQ_KERNEL_DONE) {
        gdev->kernel_completed = 1;
        dev_info(&gdev->pdev->dev, "Kernel execution completed\n");
    }

    // 处理 DMA 完成中断
    if (irq_status & GPGPU_IRQ_DMA_DONE) {
        gdev->dma_in_progress = false;
        dev_info(&gdev->pdev->dev, "DMA transfer completed\n");
    }

    // 处理错误中断
    if (irq_status & GPGPU_IRQ_ERROR) {
        gdev->error_occurred = 1;
        dev_err(&gdev->pdev->dev, "Error interrupt received\n");
    }

    // 唤醒等待队列
    wake_up(&gdev->kernel_wq);

    // 清除中断状态（写1清除）
    gpgpu_writel(gdev, GPGPU_REG_IRQ_STATUS, irq_status);
    // 确认中断
    gpgpu_writel(gdev, GPGPU_REG_IRQ_ACK, 1);

    return IRQ_HANDLED;
}

static long gpgpu_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct gpgpu_dev *gdev = filp->private_data;
    u32 grid[3], block[3];
    u32 status;
    int ret = 0;

    if (!gdev || !gdev->pdev)
        return -ENODEV;

    switch (cmd) {
    case GPGPU_IOCTL_SET_GRID_DIM:
        if (copy_from_user(grid, (void __user *)arg, sizeof(grid)))
            return -EFAULT;
        gpgpu_writel(gdev, GPGPU_REG_GRID_DIM_X, grid[0]);
        gpgpu_writel(gdev, GPGPU_REG_GRID_DIM_Y, grid[1]);
        gpgpu_writel(gdev, GPGPU_REG_GRID_DIM_Z, grid[2]);
        dev_dbg(&gdev->pdev->dev, "Grid dim: %u x %u x %u\n",
                grid[0], grid[1], grid[2]);
        break;

    case GPGPU_IOCTL_SET_BLOCK_DIM:
        if (copy_from_user(block, (void __user *)arg, sizeof(block)))
            return -EFAULT;
        gpgpu_writel(gdev, GPGPU_REG_BLOCK_DIM_X, block[0]);
        gpgpu_writel(gdev, GPGPU_REG_BLOCK_DIM_Y, block[1]);
        gpgpu_writel(gdev, GPGPU_REG_BLOCK_DIM_Z, block[2]);
        dev_dbg(&gdev->pdev->dev, "Block dim: %u x %u x %u\n",
                block[0], block[1], block[2]);
        break;

    case GPGPU_IOCTL_LAUNCH_KERNEL:
        // 检查设备是否就绪
        status = gpgpu_readl(gdev, GPGPU_REG_GLOBAL_STATUS);
        if (!(status & GPGPU_STATUS_READY)) {
            dev_warn(&gdev->pdev->dev, "Device not ready, status=0x%x\n", status);
            return -EBUSY;
        }
        // 重置完成标志
        gdev->kernel_completed = 0;
        gdev->error_occurred = 0;
        // 启动内核
        gpgpu_writel(gdev, GPGPU_REG_DISPATCH, 1);
        dev_info(&gdev->pdev->dev, "Kernel launched\n");
        break;

    case GPGPU_IOCTL_WAIT_KERNEL:
        if (gdev->irq_enabled) {
            // 有中断支持：等待中断唤醒，或超时（5秒）
            ret = wait_event_interruptible_timeout(gdev->kernel_wq,
                                                   gdev->kernel_completed || gdev->error_occurred,
                                                   5 * HZ);
            if (ret == 0) {
                dev_err(&gdev->pdev->dev, "Kernel wait timeout\n");
                return -ETIMEDOUT;
            }
            if (ret < 0) {
                dev_err(&gdev->pdev->dev, "Wait interrupted: %d\n", ret);
                return -ERESTARTSYS;
            }
        } else {
            // 无中断支持：轮询模式
            int timeout = 5000000;  // 5秒超时（微秒）
            while (timeout-- > 0) {
                status = gpgpu_readl(gdev, GPGPU_REG_GLOBAL_STATUS);
                if (!(status & GPGPU_STATUS_BUSY)) {
                    break;
                }
                udelay(1);  // 等待1微秒
            }
            if (timeout <= 0) {
                dev_err(&gdev->pdev->dev, "Kernel poll timeout\n");
                return -ETIMEDOUT;
            }
        }
        
        if (gdev->error_occurred) {
            dev_err(&gdev->pdev->dev, "Kernel execution error\n");
            gdev->error_occurred = 0;
            return -EIO;
        }
        // 读取最终状态
        status = gpgpu_readl(gdev, GPGPU_REG_GLOBAL_STATUS);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        dev_info(&gdev->pdev->dev, "Kernel completed, status=0x%08x\n", status);
        break;

    case GPGPU_IOCTL_GET_STATUS:
        status = gpgpu_readl(gdev, GPGPU_REG_GLOBAL_STATUS);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        break;

    case GPGPU_IOCTL_GET_ERROR:
        status = gpgpu_readl(gdev, GPGPU_REG_ERROR_STATUS);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        // 读后清除错误
        gpgpu_writel(gdev, GPGPU_REG_ERROR_STATUS, status);
        break;

    case GPGPU_IOCTL_DMA_XFER:
    {
        struct gpgpu_dma_params params;
        u32 dma_status;
        int timeout;

        if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
            return -EFAULT;

        // 检查是否已有 DMA 在进行
        dma_status = gpgpu_readl(gdev, GPGPU_REG_DMA_STATUS);
        if (dma_status & GPGPU_DMA_BUSY) {
            dev_warn(&gdev->pdev->dev, "DMA already busy\n");
            return -EBUSY;
        }

        dev_info(&gdev->pdev->dev, "DMA: src=0x%llx, dst=0x%llx, size=%u\n",
                 params.src_addr, params.dst_addr, params.size);

        // 设置 DMA 参数
        gpgpu_writel(gdev, GPGPU_REG_DMA_SRC_LO, lower_32_bits(params.src_addr));
        gpgpu_writel(gdev, GPGPU_REG_DMA_SRC_HI, upper_32_bits(params.src_addr));
        gpgpu_writel(gdev, GPGPU_REG_DMA_DST_LO, lower_32_bits(params.dst_addr));
        gpgpu_writel(gdev, GPGPU_REG_DMA_DST_HI, upper_32_bits(params.dst_addr));
        gpgpu_writel(gdev, GPGPU_REG_DMA_SIZE, params.size);

        // 设置控制寄存器：启动 + 方向 + 中断使能
        u32 ctrl = GPGPU_DMA_START;
        if (params.flags & GPGPU_DMA_DIR_FROM_VRAM)
            ctrl |= GPGPU_DMA_DIR_FROM_VRAM;
        if (params.flags & GPGPU_DMA_IRQ_ENABLE)
            ctrl |= GPGPU_DMA_IRQ_ENABLE;

        gdev->dma_in_progress = true;
        gpgpu_writel(gdev, GPGPU_REG_DMA_CTRL, ctrl);

        // 如果启用了中断且有中断支持，等待中断完成
        if ((ctrl & GPGPU_DMA_IRQ_ENABLE) && gdev->irq_enabled) {
            ret = wait_event_interruptible_timeout(gdev->kernel_wq,
                                                   !gdev->dma_in_progress,
                                                   5 * HZ);
            if (ret == 0) {
                dev_err(&gdev->pdev->dev, "DMA wait timeout\n");
                return -ETIMEDOUT;
            }
        } else {
            // 轮询等待完成（无中断支持或中断被禁用）
            timeout = 1000000;  // 最多等待 1M 次
            while (timeout-- > 0) {
                dma_status = gpgpu_readl(gdev, GPGPU_REG_DMA_STATUS);
                if (!(dma_status & GPGPU_DMA_BUSY))
                    break;
                cpu_relax();
            }
            if (timeout <= 0) {
                dev_err(&gdev->pdev->dev, "DMA poll timeout\n");
                return -ETIMEDOUT;
            }
        }

        // 检查 DMA 错误
        dma_status = gpgpu_readl(gdev, GPGPU_REG_DMA_STATUS);
        if (dma_status & GPGPU_DMA_ERROR) {
            dev_err(&gdev->pdev->dev, "DMA error occurred\n");
            return -EIO;
        }

        dev_info(&gdev->pdev->dev, "DMA completed successfully\n");
        break;
    }

    case GPGPU_IOCTL_RESET:
        // 软复位设备
        gpgpu_writel(gdev, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_RESET);
        msleep(10);
        dev_info(&gdev->pdev->dev, "Device reset\n");
        break;

    default:
        dev_dbg(&gdev->pdev->dev, "Unknown ioctl cmd=0x%x\n", cmd);
        return -ENOTTY;
    }

    return 0;
}

static const struct file_operations gpgpu_fops = {
    .owner          = THIS_MODULE,
    .open           = gpgpu_open,
    .release        = gpgpu_release,
    .mmap           = gpgpu_mmap,
    .unlocked_ioctl = gpgpu_ioctl,
};

/* ========== 资源清理辅助函数 ========== */

void gpgpu_cdev_cleanup(struct gpgpu_dev *gdev)
{
    if (!gdev)
        return;
    
    if (gdev->device) {
        device_destroy(gpgpu_class, MKDEV(MAJOR(gpgpu_devt), gdev->dev_id));
        gdev->device = NULL;
    }
    
    if (gdev->cdev_created) {
        cdev_del(&gdev->cdev);
        gdev->cdev_created = false;
    }
    
    if (gdev->dev_id >= 0) {
        gpgpu_free_dev_id(gdev->dev_id);  // 替换 ida_free
        gdev->dev_id = -1;
    }
}

/* ========== PCI 驱动 ========== */

static int gpgpu_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct gpgpu_dev *gdev;
    int ret;
    dev_t devt;

    dev_info(&pdev->dev, "GPGPU device found! (%04x:%04x)\n",
             pdev->vendor, pdev->device);

    // 分配设备结构体
    gdev = devm_kzalloc(&pdev->dev, sizeof(*gdev), GFP_KERNEL);
    if (!gdev)
        return -ENOMEM;

    gdev->pdev = pdev;
    gdev->dev_id = -1;
    pci_set_drvdata(pdev, gdev);

    // 分配设备ID
    ret = gpgpu_alloc_dev_id();  // 替换 ida_alloc
    if (ret < 0) {
        dev_err(&pdev->dev, "Failed to allocate device ID\n");
        return ret;
    }
    gdev->dev_id = ret;

    // 启用 PCI 设备
    ret = pcim_enable_device(pdev);
    if (ret) {
        dev_err(&pdev->dev, "Failed to enable PCI device\n");
        goto err_free_id;
    }

    // 设置 DMA 掩码
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (ret) {
        ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
        if (ret) {
            dev_err(&pdev->dev, "Failed to set DMA mask\n");
            goto err_free_id;
        }
        dev_info(&pdev->dev, "Using 32-bit DMA mask\n");
    } else {
        dev_info(&pdev->dev, "Using 64-bit DMA mask\n");
    }

    pci_set_master(pdev);

    // 映射 BAR0 (控制寄存器)
    ret = pcim_iomap_regions(pdev, BIT(GPGPU_BAR_CTRL), "gpgpu");
    if (ret) {
        dev_err(&pdev->dev, "Failed to map BAR0\n");
        goto err_free_id;
    }
    gdev->bar0 = pcim_iomap_table(pdev)[GPGPU_BAR_CTRL];

    // 映射 BAR2 (VRAM)
    ret = pcim_iomap_regions(pdev, BIT(GPGPU_BAR_VRAM), "gpgpu");
    if (ret) {
        dev_err(&pdev->dev, "Failed to map BAR2\n");
        goto err_free_id;
    }
    gdev->bar2 = pcim_iomap_table(pdev)[GPGPU_BAR_VRAM];
    gdev->bar2_size = pci_resource_len(pdev, GPGPU_BAR_VRAM);
    
    dev_info(&pdev->dev, "BAR2 (VRAM) size: %llu MiB\n",
             (u64)gdev->bar2_size / (1024 * 1024));

    // 映射 BAR4 (门铃寄存器)
    ret = pcim_iomap_regions(pdev, BIT(GPGPU_BAR_DOORBELL), "gpgpu");
    if (ret) {
        dev_err(&pdev->dev, "Failed to map BAR4\n");
        goto err_free_id;
    }
    gdev->bar4 = pcim_iomap_table(pdev)[GPGPU_BAR_DOORBELL];

    // ========== 中断初始化 ==========
    
    // 初始化等待队列
    init_waitqueue_head(&gdev->kernel_wq);
    gdev->kernel_completed = 0;
    gdev->error_occurred = 0;
    gdev->dma_in_progress = false;
    gdev->irq_enabled = false;

    // 1. 优先尝试 MSI (通常单队列设备首选)
    ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI);
    if (ret < 0) {
        dev_warn(&pdev->dev, "MSI failed (%d), trying MSI-X\n", ret);
        
        // 2. 尝试 MSI-X (如果你需要多队列)
        ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSIX);
        if (ret < 0) {
            dev_warn(&pdev->dev, "MSI-X failed (%d), falling back to Legacy INTx\n", ret);
            
            // 3. 最后尝试传统 INTx (flags = 0)
            ret = pci_alloc_irq_vectors(pdev, 1, 1, 0);
            if (ret < 0) {
                dev_warn(&pdev->dev, "All IRQ types failed: %d, continuing without interrupts\n", ret);
                dev_warn(&pdev->dev, "Device will work in polling mode only\n");
                gdev->irq = 0;
                gdev->irq_enabled = false;
            } else {
                gdev->irq = pci_irq_vector(pdev, 0);
                dev_info(&pdev->dev, "Using IRQ %d\n", gdev->irq);

                // 注册中断处理函数
                ret = request_irq(gdev->irq, gpgpu_irq_handler, 0, "gpgpu", gdev);
                if (ret) {
                    dev_err(&pdev->dev, "Failed to request IRQ %d: %d\n", gdev->irq, ret);
                    pci_free_irq_vectors(pdev);
                    goto err_free_id;
                }
                gdev->irq_enabled = true;
            }
        } else {
            gdev->irq = pci_irq_vector(pdev, 0);
            dev_info(&pdev->dev, "Using IRQ %d\n", gdev->irq);

            // 注册中断处理函数
            ret = request_irq(gdev->irq, gpgpu_irq_handler, 0, "gpgpu", gdev);
            if (ret) {
                dev_err(&pdev->dev, "Failed to request IRQ %d: %d\n", gdev->irq, ret);
                pci_free_irq_vectors(pdev);
                goto err_free_id;
            }
            gdev->irq_enabled = true;
        }
    } else {
        gdev->irq = pci_irq_vector(pdev, 0);
        dev_info(&pdev->dev, "Using IRQ %d\n", gdev->irq);

        // 注册中断处理函数
        ret = request_irq(gdev->irq, gpgpu_irq_handler, 0, "gpgpu", gdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to request IRQ %d: %d\n", gdev->irq, ret);
            pci_free_irq_vectors(pdev);
            goto err_free_id;
        }
        gdev->irq_enabled = true;
    }

    // 启用设备中断（在硬件寄存器中）
    gpgpu_writel(gdev, GPGPU_REG_IRQ_ENABLE, 
                 GPGPU_IRQ_KERNEL_DONE | GPGPU_IRQ_DMA_DONE | GPGPU_IRQ_ERROR);

    // 验证设备 ID 寄存器
    u32 dev_id_val = gpgpu_readl(gdev, GPGPU_REG_DEV_ID);
    if (dev_id_val != GPGPU_DEV_ID_VALUE) {
        dev_warn(&pdev->dev, "Unexpected DEV_ID: 0x%08x (expected 0x%08x)\n",
                 dev_id_val, GPGPU_DEV_ID_VALUE);
    }

    // ========== 字符设备初始化 ==========
    
    devt = MKDEV(MAJOR(gpgpu_devt), gdev->dev_id);
    
    // 初始化 cdev
    cdev_init(&gdev->cdev, &gpgpu_fops);
    gdev->cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&gdev->cdev, devt, 1);
    if (ret) {
        dev_err(&pdev->dev, "Failed to add cdev\n");
        goto err_free_id;
    }
    gdev->cdev_created = true;

    // 创建设备节点
    gdev->device = device_create(gpgpu_class, &pdev->dev, devt, gdev,
                                "gpgpu%d", gdev->dev_id);
    if (IS_ERR(gdev->device)) {
        dev_err(&pdev->dev, "Failed to create device node\n");
        ret = PTR_ERR(gdev->device);
        goto err_cdev;
    }

    dev_info(&pdev->dev, "Character device created: /dev/gpgpu%d\n", 
             gdev->dev_id);
    dev_info(&pdev->dev, "GPGPU driver initialized successfully\n");

    return 0;

err_cdev:
    cdev_del(&gdev->cdev);
    gdev->cdev_created = false;
err_free_id:
    gpgpu_free_dev_id(gdev->dev_id);  // 替换 ida_free
    return ret;
}

static void gpgpu_remove(struct pci_dev *pdev)
{
    struct gpgpu_dev *gdev = pci_get_drvdata(pdev);
    
    if (!gdev)
        return;

    // 禁用设备中断
    if (gdev->bar0)
        gpgpu_writel(gdev, GPGPU_REG_IRQ_ENABLE, 0);

    // 释放中断
    if (gdev->irq_enabled && gdev->irq) {
        free_irq(gdev->irq, gdev);
        pci_free_irq_vectors(pdev);
    }

    // 清理字符设备
    gpgpu_cdev_cleanup(gdev);
    
    dev_info(&pdev->dev, "Device removed\n");
}

static struct pci_device_id gpgpu_pci_ids[] = {
    { PCI_DEVICE(GPGPU_VENDOR_ID, GPGPU_DEVICE_ID) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, gpgpu_pci_ids);

static struct pci_driver gpgpu_pci_driver = {
    .name     = "gpgpu",
    .id_table = gpgpu_pci_ids,
    .probe    = gpgpu_probe,
    .remove   = gpgpu_remove,
};

/* ========== 模块初始化和退出 ========== */

static int __init gpgpu_init(void)
{
    int ret;
    
    // 分配字符设备号区域
    ret = alloc_chrdev_region(&gpgpu_devt, 0, GPGPU_MAX_DEVICES, "gpgpu");
    if (ret) {
        pr_err("gpgpu: Failed to alloc chrdev region\n");
        return ret;
    }
    
    // 创建类
    gpgpu_class = class_create("gpgpu");
    if (IS_ERR(gpgpu_class)) {
        pr_err("gpgpu: Failed to create class\n");
        ret = PTR_ERR(gpgpu_class);
        goto err_unregister;
    }
    
    // 注册PCI驱动
    ret = pci_register_driver(&gpgpu_pci_driver);
    if (ret) {
        pr_err("gpgpu: Failed to register PCI driver\n");
        goto err_class;
    }
    
    pr_info("gpgpu: Driver loaded successfully\n");
    return 0;

err_class:
    class_destroy(gpgpu_class);
err_unregister:
    unregister_chrdev_region(gpgpu_devt, GPGPU_MAX_DEVICES);
    return ret;
}

static void __exit gpgpu_exit(void)
{
    pci_unregister_driver(&gpgpu_pci_driver);
    class_destroy(gpgpu_class);
    unregister_chrdev_region(gpgpu_devt, GPGPU_MAX_DEVICES);
    // 移除 ida_destroy(&gpgpu_ida);
    
    pr_info("gpgpu: Driver unloaded\n");
}

module_init(gpgpu_init);
module_exit(gpgpu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("GPGPU PCI Device Driver");