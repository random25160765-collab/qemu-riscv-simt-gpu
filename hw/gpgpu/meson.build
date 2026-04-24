# GPGPU 设备的 Meson 构建配置

vortex_home = get_option('vortex_home')
if vortex_home == ''
  vortex_home = '/home/rd/courses/vortex'
endif

sf_lib = vortex_home / 'third_party/softfloat/build/Linux-x86_64-GCC/softfloat.a'
simx_dir = meson.current_source_dir() / 'simx'

# --- 用 make 编译 libsimx.a ---
libsimx = custom_target('libsimx',
  output: 'libsimx.a',
  input: [],
  command: [
    'make', '-f', simx_dir / 'Makefile.simx',
    'VORTEX_HOME=' + vortex_home,
    'DESTDIR=' + meson.current_build_dir() + '/',
  ],
  build_by_default: false,
  build_always_stale: false,
)

# bridge 的 include 路径
simx_incdirs = include_directories(
  'simx',
  vortex_home / 'sim/simx',
  vortex_home / 'sim/common',
  vortex_home / 'third_party/softfloat/source/include',
)

simx_dep = declare_dependency(
  sources: libsimx,
  link_args: [
    meson.current_build_dir() / 'libsimx.a',
    sf_lib,
    '-lstdc++',
  ],
  include_directories: simx_incdirs,
)

# 当 CONFIG_GPGPU 配置选项被启用时，编译核心设备文件
system_ss.add(when: 'CONFIG_GPGPU', if_true: files(
    'gpgpu.c',
    'gpgpu_core.c',
    'gpgpu_log.c',
    'lpfp.c',
    'memory.c',
))

# CONFIG_GPGPU_SIMX：启用 SimX 后端
# 启用时链接真正的 SimX bridge + libsimx.a；
# 否则链接 null stub，vx_bridge_create 始终返回 NULL，走内置解释器
system_ss.add(when: 'CONFIG_GPGPU_SIMX',
  if_true: [files('vortex_bridge.cpp'), simx_dep],
  if_false: files('vortex_bridge_null.c'),
)
