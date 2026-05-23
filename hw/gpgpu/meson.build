# GPGPU 设备的 Meson 构建配置
#
# VPU 核心 (vpu/core/*.c) 已独立编译为 libvpu.a，不再通过 QEMU 构建系统编译。
# QEMU 仅保留 gpgpu.c 作为薄管道层 (PCI BAR + IRQ 投递)。

# 当 CONFIG_GPGPU 配置选项被启用时，编译 gpgpu.c (QEMU 前端)
system_ss.add(when: 'CONFIG_GPGPU', if_true: files(
    'gpgpu.c',
))
