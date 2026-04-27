#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SimX 后端句柄，由 vx_bridge_create 返回 */
typedef struct VxBridgeHandle VxBridgeHandle;

/* 创建 SimX 后端实例（每个 GPGPU 设备一个） */
VxBridgeHandle *vx_bridge_create(uint32_t num_cores,
                                  uint32_t num_warps,
                                  uint32_t num_threads);

/* 释放 SimX 后端实例 */
void vx_bridge_destroy(VxBridgeHandle *h);

/*
 * 执行一个 kernel。
 *   vram        : 整块 VRAM 的宿主机指针
 *   vram_size   : VRAM 字节数
 *   kernel_addr : kernel 二进制在 VRAM 内的偏移（同时作为 startup_addr）
 * 返回 0 表示成功，非零表示 kernel 返回了非零 exitcode。
 */
int vx_bridge_run(VxBridgeHandle *h,
                  const uint8_t  *vram,
                  uint64_t        vram_size,
                  uint64_t        kernel_addr,
                  const uint32_t  grid_dim[3],
                  const uint32_t  block_dim[3]);

#ifdef __cplusplus
}
#endif
