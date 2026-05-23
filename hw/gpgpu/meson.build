# GPGPU 设备的 Meson 构建配置

# 当 CONFIG_GPGPU 配置选项被启用时，编译 gpgpu.c (QEMU 前端)
system_ss.add(when: 'CONFIG_GPGPU', if_true: files(
    'gpgpu.c',
))
