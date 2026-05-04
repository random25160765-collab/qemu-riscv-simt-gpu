/*
 * test_e2e_sim.c — SIM 模式下端到端调用链冒烟测试
 *
 * 模拟 OpenCL vector_add 的完整运行时调用路径，但不依赖 PoCL 头：
 *   device init  →  alloc buffers  →  mem_write inputs
 *               →  kernel create + set_arg + work_dim
 *               →  launch_kernel + wait_kernel  (submit 的核心)
 *               →  mem_read result  →  destroy
 *
 * SIM 模式下 mem_write/read 是 no-op，launch_kernel 是 no-op；
 * 这个测试验证的是 *调用契约* 端到端不会崩、状态机正确。
 */

#include "conflux_device.h"
#include "conflux_platform.h"
#include "conflux_kernel.h"
#include "conflux_allocator.h"
#include "conflux_hal.h"
#include "conflux_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define N 256

/* dummy kernel binary（真实场景由 LLVM 后端生成，这里只是占位字节） */
static uint8_t dummy_binary[] = {
    /* 16 字节假指令，足以让 binary_size > 0 通过校验 */
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
};

#define OK(expr) do {                                       \
    int _r = (expr);                                        \
    if (_r != CONFLUX_SUCCESS) {                            \
        fprintf(stderr, "FAIL %s:%d: %s -> %d\n",           \
                __FILE__, __LINE__, #expr, _r);             \
        return 1;                                           \
    }                                                       \
} while (0)

/*
 * 模拟 pocl_conflux_submit 在 NDRANGE_KERNEL 时的核心步骤。
 * 不依赖 PoCL 类型，直接对 conflux_kernel_t 操作。
 *
 * 这是 pocl_conflux_ops.c:pocl_conflux_submit 的"运行时层"等价实现，
 * 抽离出来便于在没有 PoCL 头时也能验证。
 */
static int sim_submit_kernel(conflux_device_t *dev,
                             conflux_kernel_t *kernel)
{
    conflux_allocator_t *alloc = conflux_device_get_allocator(dev);
    if (!alloc) return -1;

    /* 1) binary 上传 */
    if (!kernel->binary_uploaded) {
        uint64_t addr = conflux_allocator_alloc(alloc, kernel->binary_size);
        if (addr == UINT64_MAX) return -2;
        if (conflux_hal_mem_write(&dev->hal, addr,
                                  kernel->binary, kernel->binary_size)
            != CONFLUX_SUCCESS) return -3;
        kernel->binary_device_addr = addr;
        kernel->binary_uploaded    = 1;
    }

    /* 2) args 紧密打包到 VRAM */
    size_t args_total = 0;
    for (uint32_t i = 0; i < kernel->num_args; i++) {
        args_total += kernel->args[i].size;
    }
    uint64_t args_addr = 0;
    if (args_total > 0) {
        uint8_t *packed = (uint8_t *)malloc(args_total);
        if (!packed) return -4;
        size_t off = 0;
        for (uint32_t i = 0; i < kernel->num_args; i++) {
            conflux_kernel_arg_t *a = &kernel->args[i];
            if (a->value && a->size > 0) {
                memcpy(packed + off, a->value, a->size);
            }
            off += a->size;
        }
        args_addr = conflux_allocator_alloc(alloc, args_total);
        if (args_addr == UINT64_MAX) { free(packed); return -5; }
        int wr = conflux_hal_mem_write(&dev->hal, args_addr,
                                       packed, args_total);
        free(packed);
        if (wr != CONFLUX_SUCCESS) return -6;
    }

    /* 3) grid/block from work_dim */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    for (uint32_t d = 0; d < kernel->work_dim && d < 3; d++) {
        size_t lws = kernel->local_size[d] ? kernel->local_size[d] : 1;
        block[d]   = (uint32_t)lws;
        grid[d]    = (uint32_t)((kernel->global_size[d] + lws - 1) / lws);
    }

    /* 4) launch + wait */
    int lr = conflux_hal_launch_kernel(&dev->hal,
                                       kernel->binary_device_addr,
                                       args_addr, grid, block,
                                       /*shared_mem*/ 0);
    if (lr != CONFLUX_SUCCESS) return -7;
    int wr = conflux_hal_wait_kernel(&dev->hal, /*timeout*/ 0);
    if (wr != CONFLUX_SUCCESS) return -8;
    return 0;
}

int main(void)
{
    printf("=== test_e2e_sim: vector_add path ===\n\n");
    conflux_log_init(CONFLUX_LOG_INFO);

    /* ===== 设备生命周期（测试专用：显式注入 SIM 描述符，不走 probe） ===== */
    printf("--- 1. inject SIM device + open ---\n");
    OK(conflux_platform_init());

    /* 测试模式：用 probe_specific 注入 path="" 的描述符 → SIM 模式
     * 生产代码应该调用 conflux_platform_probe()，要求 /dev/gpgpu0 已存在 */
    conflux_device_desc_t sim_desc;
    memset(&sim_desc, 0, sizeof(sim_desc));
    sim_desc.dev_index = 0;
    sim_desc.vendor_id = CONFLUX_DEVICE_VENDOR_ID;
    sim_desc.device_id = CONFLUX_DEVICE_DEVICE_ID;
    sim_desc.mmio_base = 0x10000000;
    sim_desc.mmio_size = 64ULL * 1024 * 1024;
    sim_desc.available = 1;
    sim_desc.path[0]   = '\0';   /* 空路径 → SIM 模式 */
    int idx = conflux_platform_probe_specific(&sim_desc);
    assert(idx == 0);

    OK(conflux_platform_open_device(0));
    conflux_device_t *dev = conflux_platform_get_device(0);
    assert(dev != NULL);
    assert(conflux_device_is_online(dev));
    assert(dev->hal.mode == CONFLUX_HAL_MODE_SIM);
    printf("  device=%s, hal_mode=%d (SIM)\n",
           dev->name, dev->hal.mode);
    printf("  OK\n\n");

    /* ===== 缓冲区分配（模拟 clCreateBuffer × 3） ===== */
    printf("--- 2. allocate A, B, C buffers (N=%d, %zu bytes each) ---\n",
           N, N * sizeof(float));
    conflux_allocator_t *alloc = conflux_device_get_allocator(dev);
    assert(alloc);
    uint64_t buf_a = conflux_allocator_alloc(alloc, N * sizeof(float));
    uint64_t buf_b = conflux_allocator_alloc(alloc, N * sizeof(float));
    uint64_t buf_c = conflux_allocator_alloc(alloc, N * sizeof(float));
    assert(buf_a != UINT64_MAX && buf_b != UINT64_MAX && buf_c != UINT64_MAX);
    printf("  A=0x%lx  B=0x%lx  C=0x%lx\n",
           (unsigned long)buf_a, (unsigned long)buf_b, (unsigned long)buf_c);
    printf("  OK\n\n");

    /* ===== 上传输入数据（模拟 clEnqueueWriteBuffer × 2） ===== */
    printf("--- 3. mem_write A, B (host -> device) ---\n");
    float host_a[N], host_b[N], host_c[N];
    for (int i = 0; i < N; i++) {
        host_a[i] = (float)i;
        host_b[i] = (float)(i * 2);
        host_c[i] = -1.0f;  /* 哨兵 */
    }
    OK(conflux_hal_mem_write(&dev->hal, buf_a, host_a, sizeof(host_a)));
    OK(conflux_hal_mem_write(&dev->hal, buf_b, host_b, sizeof(host_b)));
    printf("  OK\n\n");

    /* ===== 创建 kernel + set_arg ===== */
    printf("--- 4. kernel create + set_args ---\n");
    conflux_kernel_t *k = conflux_kernel_create("vector_add",
                                                /*kernel_id*/ 1,
                                                dummy_binary,
                                                sizeof(dummy_binary));
    assert(k);

    OK(conflux_kernel_set_arg(k, 0, sizeof(uint64_t), &buf_a, /*is_local*/ 0));
    OK(conflux_kernel_set_arg(k, 1, sizeof(uint64_t), &buf_b, 0));
    OK(conflux_kernel_set_arg(k, 2, sizeof(uint64_t), &buf_c, 0));
    int n = N;
    OK(conflux_kernel_set_arg(k, 3, sizeof(int), &n, 0));
    assert(k->num_args == 4);

    OK(conflux_kernel_set_work_dim(k, 1));
    OK(conflux_kernel_set_global_size(k, 0, N));
    OK(conflux_kernel_set_local_size(k, 0, 64));
    printf("  num_args=%u, global=%zu, local=%zu\n",
           k->num_args, k->global_size[0], k->local_size[0]);
    printf("  OK\n\n");

    /* ===== submit（同步路径） ===== */
    printf("--- 5. submit (mimics pocl_conflux_submit) ---\n");
    int sret = sim_submit_kernel(dev, k);
    if (sret != 0) {
        fprintf(stderr, "FAIL sim_submit_kernel -> %d\n", sret);
        return 1;
    }
    assert(k->binary_uploaded);
    printf("  binary uploaded at 0x%lx\n",
           (unsigned long)k->binary_device_addr);
    printf("  OK\n\n");

    /* ===== 重复 submit：binary 不应再次上传 ===== */
    printf("--- 6. second submit reuses uploaded binary ---\n");
    uint64_t prev_addr = k->binary_device_addr;
    sret = sim_submit_kernel(dev, k);
    assert(sret == 0);
    assert(k->binary_device_addr == prev_addr);
    printf("  OK\n\n");

    /* ===== 读回结果（模拟 clEnqueueReadBuffer） ===== */
    printf("--- 7. mem_read C (device -> host) ---\n");
    OK(conflux_hal_mem_read(&dev->hal, buf_c, host_c, sizeof(host_c)));
    /* SIM 模式：mem_read 把 buf 清零（no real device） */
    for (int i = 0; i < N; i++) assert(host_c[i] == 0.0f);
    printf("  OK (SIM zeroed result, as documented)\n\n");

    /* ===== 清理 ===== */
    printf("--- 8. cleanup ---\n");
    conflux_kernel_destroy(k);
    conflux_platform_close_device(0);
    conflux_platform_destroy();
    printf("  OK\n\n");

    printf("=== test_e2e_sim: ALL PASSED ===\n");
    return 0;
}
