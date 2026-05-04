/* test_dma.c — 直接用 conflux API 验证 DMA 读写路径
 *
 * 不依赖 PoCL/OpenCL，直接测：
 *   conflux_hal_mem_write → ioctl(DMA_XFER) → QEMU VRAM
 *   conflux_hal_mem_read  → ioctl(DMA_XFER) → host
 *
 * 编译（guest 里）：
 *   gcc test_dma.c -I/mnt/hostshare/conflux-runtime/include \
 *       -L/usr/local/lib -lconflux -lpthread -o /tmp/test_dma
 *
 * 运行：
 *   LD_LIBRARY_PATH=/usr/local/lib /tmp/test_dma
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "conflux_platform.h"
#include "conflux_device.h"
#include "conflux_hal.h"
#include "conflux_allocator.h"
#include "conflux_log.h"

#define N 256  /* 256 floats = 1024 bytes，刚好一页以内 */

int main(void)
{
    conflux_log_init(CONFLUX_LOG_ERROR);  /* 只打 error，减少噪音 */

    printf("=== DMA 读写测试 ===\n\n");

    /* 1. 初始化平台/设备 */
    assert(conflux_platform_init() == CONFLUX_SUCCESS);
    int n = conflux_platform_probe();
    printf("probe: %d device(s)\n", n);
    assert(n >= 1);
    assert(conflux_platform_open_device(0) == CONFLUX_SUCCESS);

    conflux_device_t *dev = conflux_platform_get_device(0);
    assert(dev != NULL);
    printf("device: %s (HAL mode=%d)\n\n", dev->name, dev->hal.mode);

    /* 2. 分配 VRAM */
    conflux_allocator_t *alloc = conflux_device_get_allocator(dev);
    uint64_t vram_addr = conflux_allocator_alloc(alloc, N * sizeof(float));
    assert(vram_addr != UINT64_MAX);
    printf("VRAM alloc: 0x%lx\n", (unsigned long)vram_addr);

    /* 3. 分配 page-aligned 的 host 数据（驱动只 pin 一页，必须 page-aligned） */
    size_t buf_size = N * sizeof(float);
    float *h_src = aligned_alloc(4096, (buf_size + 4095) & ~4095UL);
    float *h_dst = aligned_alloc(4096, (buf_size + 4095) & ~4095UL);
    if (!h_src || !h_dst) { fprintf(stderr, "aligned_alloc failed\n"); return 1; }
    for (int i = 0; i < N; i++) h_src[i] = (float)i * 3.14f;
    memset(h_dst, 0, buf_size);

    /* 4. host → VRAM (DMA write) */
    int r = conflux_hal_mem_write(&dev->hal, vram_addr, h_src, buf_size);
    if (r != CONFLUX_SUCCESS) {
        printf("FAIL mem_write: %d\n", r);
        return 1;
    }
    printf("mem_write OK (%d floats, %zu bytes)\n", N, N * sizeof(float));

    /* 5. VRAM → host (DMA read) */
    r = conflux_hal_mem_read(&dev->hal, vram_addr, h_dst, buf_size);
    if (r != CONFLUX_SUCCESS) {
        printf("FAIL mem_read: %d\n", r);
        return 1;
    }
    printf("mem_read  OK (%d floats, %zu bytes)\n", N, N * sizeof(float));

    /* 6. 验证 */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_dst[i] != h_src[i]) {
            printf("MISMATCH [%d]: wrote %.3f, read %.3f\n",
                   i, h_src[i], h_dst[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("\nPASS: DMA round-trip OK, all %d values match\n", N);
    } else {
        printf("\nNOTE: 数据不一致。\n");
        printf("  如果 QEMU 后端是 SimX，mem_write 是 DMA 写入 VRAM，\n");
        printf("  mem_read 读回的是 VRAM 里的内容（应当一致）。\n");
        printf("  src[0]=%.3f dst[0]=%.3f  src[1]=%.3f dst[1]=%.3f\n",
               h_src[0], h_dst[0], h_src[1], h_dst[1]);
    }

    /* 7. 清理 */
    free(h_src);
    free(h_dst);
    conflux_allocator_free(alloc, vram_addr, N * sizeof(float));
    conflux_platform_close_device(0);
    conflux_platform_destroy();

    return pass ? 0 : 1;
}
