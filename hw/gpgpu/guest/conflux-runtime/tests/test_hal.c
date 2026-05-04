/*
 * test_hal.c — HAL 接口冒烟测试（SIM 模式）
 *
 * 目的：验证 conflux_hal_* API 在 SIM 模式下的调用契约不会崩，
 *       并验证关键改动（mem_* 重命名、launch_kernel 完整路径）。
 *
 * 不验证 IOCTL 模式语义（那需要真实驱动 + QEMU 设备）。
 */

#include "conflux_hal.h"
#include "conflux_log.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define OK(expr) do {                                       \
    int _r = (expr);                                        \
    if (_r != CONFLUX_SUCCESS) {                            \
        fprintf(stderr, "FAIL at %s:%d: %s -> %d\n",        \
                __FILE__, __LINE__, #expr, _r);             \
        return 1;                                           \
    }                                                       \
} while (0)

int main(void)
{
    printf("=== HAL smoke test (SIM mode) ===\n\n");

    conflux_log_init(CONFLUX_LOG_INFO);

    conflux_hal_t hal;

    /* T1: 初始化（SIM 模式，无 /dev 文件） */
    printf("--- T1: hal_init (SIM) ---\n");
    OK(conflux_hal_init(&hal, CONFLUX_HAL_MODE_SIM, NULL));
    assert(hal.initialized);
    assert(hal.mode == CONFLUX_HAL_MODE_SIM);
    printf("  OK\n\n");

    /* T2: 状态查询 */
    printf("--- T2: get_status / get_error ---\n");
    uint32_t status = 0, err = 0;
    OK(conflux_hal_get_status(&hal, &status));
    OK(conflux_hal_get_error(&hal, &err));
    printf("  status=0x%x, error=0x%x\n", status, err);
    /* SIM 模式下 get_status 返回 STATUS_READY */
    assert(status & CONFLUX_STATUS_READY);
    printf("  OK\n\n");

    /* T3: 数据传输（mem_write/read，原 vram_*）*/
    printf("--- T3: mem_write / mem_read (renamed from vram_*) ---\n");
    uint8_t src[64];
    uint8_t dst[64];
    for (int i = 0; i < 64; i++) src[i] = (uint8_t)i;
    memset(dst, 0xAA, sizeof(dst));
    OK(conflux_hal_mem_write(&hal, 0x1000, src, sizeof(src)));
    OK(conflux_hal_mem_read(&hal, 0x1000, dst, sizeof(dst)));
    /* SIM 模式下 mem_read 把 buf 清零（语义：无真实 VRAM） */
    for (int i = 0; i < 64; i++) assert(dst[i] == 0);
    printf("  OK (SIM zeroes read buffer as documented)\n\n");

    /* T4: 内核维度设置 */
    printf("--- T4: set_grid_dim / set_block_dim ---\n");
    OK(conflux_hal_set_grid_dim(&hal, 4, 2, 1));
    OK(conflux_hal_set_block_dim(&hal, 64, 1, 1));
    printf("  OK\n\n");

    /* T5: 组合启动（覆盖 F4 中 submit 调用的核心 API） */
    printf("--- T5: launch_kernel + wait_kernel ---\n");
    uint32_t grid[3]  = {4, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    OK(conflux_hal_launch_kernel(&hal,
                                 /*kernel_addr*/ 0x10000,
                                 /*args_addr*/   0x20000,
                                 grid, block,
                                 /*shared_mem*/  0));
    OK(conflux_hal_wait_kernel(&hal, /*timeout_ms*/ 100));
    printf("  OK\n\n");

    /* T6: DMA transfer 直接调用（mem_* 在 IOCTL 模式底下走的就是它） */
    printf("--- T6: dma_transfer (direct, SIM is no-op) ---\n");
    OK(conflux_hal_dma_transfer(&hal, 0, 0x1000, 64, true));
    OK(conflux_hal_dma_transfer(&hal, 0x1000, 0, 64, false));
    printf("  OK\n\n");

    /* T7: 关闭 */
    printf("--- T7: hal_close ---\n");
    conflux_hal_close(&hal);
    assert(!hal.initialized);
    printf("  OK\n\n");

    printf("=== HAL smoke test: ALL PASSED ===\n");
    return 0;
}
