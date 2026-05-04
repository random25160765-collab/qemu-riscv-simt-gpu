/*
 * test_device_queue.c — 验证 conflux_device_get_queue 惰性创建
 *                       和 device_execute_cmd 路由路径
 *
 * 走 SIM 模式：HAL 调用是 no-op，但队列和路由本身是真实的。
 */

#include "conflux_device.h"
#include "conflux_queue.h"
#include "conflux_log.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main(void)
{
    printf("=== test_device_queue ===\n\n");
    conflux_log_init(CONFLUX_LOG_INFO);

    /* T1: 创建并初始化设备（SIM 模式） */
    printf("--- T1: device init (SIM) ---\n");
    conflux_device_t *dev = conflux_device_create();
    assert(dev != NULL);
    int ret = conflux_device_init(dev, NULL,
                                  /*mmio_base*/ 0x10000000,
                                  /*mmio_size*/ 64ULL * 1024 * 1024);
    assert(ret == CONFLUX_SUCCESS);
    assert(conflux_device_online(dev) == CONFLUX_SUCCESS);
    /* 初始没有队列 */
    assert(dev->num_queues == 0);
    for (uint32_t i = 0; i < dev->max_queues; i++) {
        assert(dev->queues[i] == NULL);
    }
    printf("  OK\n\n");

    /* T2: 首次 get_queue 应惰性创建 */
    printf("--- T2: get_queue lazy creation ---\n");
    conflux_queue_t *q0 = conflux_device_get_queue(dev, 0);
    assert(q0 != NULL);
    assert(dev->queues[0] == q0);
    assert(dev->num_queues == 1);
    printf("  q0 = %p, num_queues = %u\n", (void *)q0, dev->num_queues);
    printf("  OK\n\n");

    /* T3: 重复 get_queue 同 index 应返回同一对象 */
    printf("--- T3: get_queue idempotent ---\n");
    conflux_queue_t *q0_again = conflux_device_get_queue(dev, 0);
    assert(q0_again == q0);
    assert(dev->num_queues == 1);
    printf("  OK\n\n");

    /* T4: 获取另一个索引 */
    printf("--- T4: second queue at index 2 ---\n");
    conflux_queue_t *q2 = conflux_device_get_queue(dev, 2);
    assert(q2 != NULL);
    assert(q2 != q0);
    assert(dev->queues[2] == q2);
    assert(dev->num_queues == 3);  /* 高水位线 */
    /* 中间索引 1 仍未创建 */
    assert(dev->queues[1] == NULL);
    printf("  OK\n\n");

    /* T5: 越界 */
    printf("--- T5: out-of-bounds returns NULL ---\n");
    conflux_queue_t *qbad = conflux_device_get_queue(dev, 999);
    assert(qbad == NULL);
    printf("  OK\n\n");

    /* T6: 提交 NOP 命令走 execute_cmd 路由 */
    printf("--- T6: submit NOP via queue (sync) ---\n");
    conflux_cmd_t nop = {.type = CONFLUX_CMD_NOP};
    int rc = conflux_queue_submit_sync(q0, &nop);
    assert(rc == CONFLUX_SUCCESS);
    printf("  OK\n\n");

    /* T7: 提交 BARRIER（device_execute_cmd 调 wait_kernel，SIM 直接成功） */
    printf("--- T7: submit BARRIER ---\n");
    conflux_cmd_t bar = {.type = CONFLUX_CMD_BARRIER};
    rc = conflux_queue_submit_sync(q0, &bar);
    assert(rc == CONFLUX_SUCCESS);
    printf("  OK\n\n");

    /* T8: 销毁设备应清理所有队列 */
    printf("--- T8: destroy cleans up queues ---\n");
    conflux_device_destroy(dev);
    /* 不能 dump 已销毁的 dev，但走到这里没崩说明清理路径无悬挂 */
    printf("  OK\n\n");

    printf("=== test_device_queue: ALL PASSED ===\n");
    return 0;
}
