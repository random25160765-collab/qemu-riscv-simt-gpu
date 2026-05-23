/*
 * QEMU Educational GPGPU Device — thin PCI frontend
 *
 * Copyright (c) 2024-2025
 *
 * Licensed under GPL v2 or later.
 *
 * QEMU is a thin pipe: PCI BAR exposure + IRQ delivery.
 * All GPU simulation (registers, VRAM, kernel dispatch, RISC-V SIMT)
 * runs in the VPU host process. Communication via shared memory + eventfd.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include <sys/mman.h>
#include <sys/eventfd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <spawn.h>

#include "gpgpu.h"

/* =========================================================================
 * QEMU-VPU IPC helpers
 * ========================================================================= */

static void gpgpu_send_cmd(GPGPUState *s, uint32_t cmd,
                           uint32_t data0, uint32_t data1)
{
    s->ctrl_ptr[VPU_CTRL_DATA_OFFSET / 4] = data0;
    s->ctrl_ptr[VPU_CTRL_DATA1_OFFSET / 4] = data1;
    smp_wmb(); /* ensure data is visible before command */
    s->ctrl_ptr[VPU_CTRL_CMD_OFFSET / 4] = cmd;
    uint64_t one = 1;
    eventfd_write(s->doorbell_fd, one);
}

static int gpgpu_wait_complete(GPGPUState *s)
{
    uint64_t val;
    if (eventfd_read(s->complete_fd, &val) < 0) {
        if (errno == EINTR)
            return gpgpu_wait_complete(s);
        return -1;
    }
    return 0;
}

/* =========================================================================
 * BAR 0: Control registers — forwarded to VPU
 * ========================================================================= */

static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    case GPGPU_REG_DEV_ID:
        return GPGPU_DEV_ID_VALUE;
    case GPGPU_REG_DEV_VERSION:
        return GPGPU_DEV_VERSION_VALUE;
    case GPGPU_REG_DEV_CAPS:
        return (s->num_cus & 0xFF) |
               ((s->warps_per_cu & 0xFF) << 8) |
               ((s->warp_size & 0xFF) << 16);
    case GPGPU_REG_VRAM_SIZE_LO:
        return 0x04000000;
    case GPGPU_REG_VRAM_SIZE_HI:
        return 0x00000000;
    case GPGPU_REG_IRQ_ENABLE:
        return s->irq_enable;
    case GPGPU_REG_IRQ_STATUS:
        return s->irq_status;
    default:
        break;
    }

    /* Forward to VPU for all other registers */
    gpgpu_send_cmd(s, VPU_CMD_REG_READ, (uint32_t)addr, 0);
    if (gpgpu_wait_complete(s) < 0)
        return ~0ULL;
    return s->ctrl_ptr[VPU_CTRL_DATA1_OFFSET / 4];
}

static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = val;
        return;
    case GPGPU_REG_IRQ_ACK:
        s->irq_status &= ~val;
        return;
    default:
        break;
    }

    /* Forward to VPU */
    gpgpu_send_cmd(s, VPU_CMD_REG_WRITE, (uint32_t)addr, (uint32_t)val);
    /* Writes are fire-and-forget (no completion wait) */
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* =========================================================================
 * BAR 2: VRAM — mapped directly from VPU shared memory
 * ========================================================================= */

/* =========================================================================
 * BAR 4: Doorbell — write triggers eventfd to VPU
 * ========================================================================= */

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    case 0x0330: /* DISPATCH */
        gpgpu_send_cmd(s, VPU_CMD_DISPATCH, 0, 0);
        break;
    default:
        break;
    }
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* =========================================================================
 * Completion eventfd handler — VPU done → deliver IRQ
 * ========================================================================= */

static void gpgpu_complete_handler(void *opaque)
{
    GPGPUState *s = opaque;
    uint64_t val;

    if (eventfd_read(s->complete_fd, &val) < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            return;
        qemu_log("GPGPU: error reading completion eventfd: %s\n",
                 strerror(errno));
        return;
    }

    /* Read kernel completion status from CTRL shm */
    int32_t ret = (int32_t)s->ctrl_ptr[VPU_CTRL_DATA_OFFSET / 4];

    if (ret == 0) {
        s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
        if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
            if (msix_enabled(&s->parent_obj))
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_KERNEL);
            else if (msi_enabled(&s->parent_obj))
                msi_notify(&s->parent_obj, GPGPU_MSIX_VEC_KERNEL);
            else
                pci_set_irq(&s->parent_obj, 1);
        }
    } else {
        s->irq_status |= GPGPU_IRQ_ERROR;
        if (s->irq_enable & GPGPU_IRQ_ERROR) {
            if (msix_enabled(&s->parent_obj))
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_ERROR);
            else if (msi_enabled(&s->parent_obj))
                msi_notify(&s->parent_obj, GPGPU_MSIX_VEC_ERROR);
            else
                pci_set_irq(&s->parent_obj, 1);
        }
    }
}

/* =========================================================================
 * VPU child process management
 * ========================================================================= */

static void gpgpu_spawn_vpu(GPGPUState *s, Error **errp)
{
    char doorbell_str[32], complete_str[32];

    snprintf(doorbell_str, sizeof(doorbell_str), "%d", s->doorbell_fd);
    snprintf(complete_str, sizeof(complete_str), "%d", s->complete_fd);

    char *argv[] = {
        (char *)"/home/rd/courses/qemu-riscv-simt-gpu/hw/gpgpu/vpu/build/vpu",
        NULL
    };

    char *envp[] = {
        g_strdup_printf("%s=%s", VPU_ENV_DOORBELL_FD, doorbell_str),
        g_strdup_printf("%s=%s", VPU_ENV_COMPLETE_FD, complete_str),
        NULL
    };

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);

    int ret = posix_spawnp(&s->vpu_pid, argv[0], &fa, NULL, argv, envp);
    posix_spawn_file_actions_destroy(&fa);

    g_free(envp[0]);
    g_free(envp[1]);

    if (ret != 0) {
        error_setg(errp, "GPGPU: failed to spawn VPU process: %s",
                   strerror(errno));
    }
}

/* =========================================================================
 * Device lifecycle
 * ========================================================================= */

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    /* Create VRAM shared memory */
    s->vram_shm_fd = shm_open(VPU_SHM_VRAM_NAME,
                              O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (s->vram_shm_fd < 0) {
        error_setg(errp, "GPGPU: failed to create VRAM shm: %s",
                   strerror(errno));
        return;
    }
    if (ftruncate(s->vram_shm_fd, s->vram_size) < 0) {
        error_setg(errp, "GPGPU: failed to set VRAM shm size: %s",
                   strerror(errno));
        close(s->vram_shm_fd);
        return;
    }
    s->vram_ptr = mmap(NULL, s->vram_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, s->vram_shm_fd, 0);
    if (s->vram_ptr == MAP_FAILED) {
        error_setg(errp, "GPGPU: failed to mmap VRAM: %s", strerror(errno));
        close(s->vram_shm_fd);
        return;
    }

    /* Create CTRL shared memory */
    s->ctrl_shm_fd = shm_open(VPU_SHM_CTRL_NAME,
                              O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (s->ctrl_shm_fd < 0) {
        error_setg(errp, "GPGPU: failed to create CTRL shm: %s",
                   strerror(errno));
        return;
    }
    if (ftruncate(s->ctrl_shm_fd, VPU_CTRL_SIZE) < 0) {
        error_setg(errp, "GPGPU: failed to set CTRL shm size: %s",
                   strerror(errno));
        return;
    }
    s->ctrl_ptr = mmap(NULL, VPU_CTRL_SIZE, PROT_READ | PROT_WRITE,
                       MAP_SHARED, s->ctrl_shm_fd, 0);
    if (s->ctrl_ptr == MAP_FAILED) {
        error_setg(errp, "GPGPU: failed to mmap CTRL: %s", strerror(errno));
        return;
    }

    /* Create eventfds */
    s->doorbell_fd = eventfd(0, EFD_SEMAPHORE);
    s->complete_fd = eventfd(0, EFD_NONBLOCK);
    if (s->doorbell_fd < 0 || s->complete_fd < 0) {
        error_setg(errp, "GPGPU: failed to create eventfd");
        return;
    }

    /* BAR 0: control registers (MMIO forwarded to VPU) */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM (mapped from VPU shared memory) */
    memory_region_init_ram_ptr(&s->vram, OBJECT(s), "gpgpu-vram",
                               s->vram_size, s->vram_ptr);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    /* MSI-X / MSI / INTx */
    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        msi_init(pdev, 0, 1, true, false, errp);
    } else {
        msi_init(pdev, 0, 1, true, false, errp);
    }

    /* Watch completion eventfd for IRQ delivery */
    qemu_set_fd_handler(s->complete_fd, gpgpu_complete_handler, NULL, s);

    /* Spawn VPU child process */
    gpgpu_spawn_vpu(s, errp);
    if (*errp)
        return;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    qemu_set_fd_handler(s->complete_fd, NULL, NULL, NULL);

    if (s->vpu_pid > 0) {
        kill(s->vpu_pid, SIGTERM);
        waitpid(s->vpu_pid, NULL, 0);
    }

    if (s->vram_ptr && s->vram_ptr != MAP_FAILED)
        munmap(s->vram_ptr, s->vram_size);
    if (s->vram_shm_fd >= 0) {
        close(s->vram_shm_fd);
        shm_unlink(VPU_SHM_VRAM_NAME);
    }

    if (s->ctrl_ptr && s->ctrl_ptr != MAP_FAILED)
        munmap(s->ctrl_ptr, VPU_CTRL_SIZE);
    if (s->ctrl_shm_fd >= 0) {
        close(s->ctrl_shm_fd);
        shm_unlink(VPU_SHM_CTRL_NAME);
    }

    if (s->doorbell_fd >= 0)
        close(s->doorbell_fd);
    if (s->complete_fd >= 0)
        close(s->complete_fd);

    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->irq_enable = 0;
    s->irq_status = 0;

    if (s->vpu_pid > 0) {
        gpgpu_send_cmd(s, VPU_CMD_RESET, 0, 0);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
