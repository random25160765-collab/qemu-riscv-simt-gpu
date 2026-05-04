# PoCL Patches for Conflux GPGPU Device

## 使用方式

在 `guest/pocl/` 目录下应用：

```bash
cd hw/gpgpu/guest/pocl
git apply ../patches/0001-add-conflux-gpgpu-device.patch
```

还原：

```bash
git apply --reverse ../patches/0001-add-conflux-gpgpu-device.patch
# 新增文件需要手动删除
rm -f lib/CL/devices/gpgpu/CMakeLists.txt lib/CL/devices/gpgpu/pocl-gpgpu.c
```

## Patch 列表

### 0001-add-conflux-gpgpu-device.patch

修改范围：

| 文件 | 修改内容 |
|:---|:---|
| `CMakeLists.txt` | 加 `set(BUILD_GPGPU ${ENABLE_GPGPU})`；修复 dlfcn 交叉编译跳过条件加 `AND (NOT ENABLE_GPGPU)` |
| `config.h.in.cmake` | 加 `#cmakedefine BUILD_GPGPU`，使 `devices.c` 能用 `#ifdef BUILD_GPGPU` 编译 gpgpu 设备 |
| `lib/CL/devices/CMakeLists.txt` | 把 `list(APPEND POCL_DEVICES_LINK_LIST gpgpu)` 改为 `find_library` 找 `libconflux`（含 `NO_CMAKE_FIND_ROOT_PATH`），移到父目录避免子目录变量作用域问题 |
| `lib/CL/devices/devices.c` | 在 `pocl_devices_init_ops[]` 和 `pocl_device_types[]` 两个数组里加 `BUILD_GPGPU` 条件分支 |
| `lib/CL/devices/gpgpu/CMakeLists.txt` | **新增**：find_library libconflux；设置 include 路径；链接 conflux + pthread |
| `lib/CL/devices/gpgpu/pocl-gpgpu.c` | **新增**：PoCL `pocl_device_ops` 回调实现，内部调用 conflux-runtime API |
