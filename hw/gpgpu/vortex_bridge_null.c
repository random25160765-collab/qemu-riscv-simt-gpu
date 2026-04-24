/* 空实现：未启用 SimX 后端时使用，所有函数返回 NULL / 无操作 */
#include "vortex_bridge.h"

VxBridgeHandle *vx_bridge_create(uint32_t num_cores,
                                  uint32_t num_warps,
                                  uint32_t num_threads)
{
    (void)num_cores; (void)num_warps; (void)num_threads;
    return NULL;
}

void vx_bridge_destroy(VxBridgeHandle *h)
{
    (void)h;
}

int vx_bridge_run(VxBridgeHandle *h,
                  const uint8_t *vram,
                  uint64_t vram_size,
                  uint64_t kernel_addr)
{
    (void)h; (void)vram; (void)vram_size; (void)kernel_addr;
    return -1;
}
