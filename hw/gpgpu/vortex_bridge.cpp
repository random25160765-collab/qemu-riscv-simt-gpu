/*
 * Vortex SimX bridge：将 QEMU GPGPU 设备的执行请求转发给 SimX。
 *
 * 接口语义：
 *   - vx_bridge_create  : 构造 Processor + RAM，绑定 DCR
 *   - vx_bridge_run     : 把 VRAM 内容写入 SimX RAM，设置 startup_addr，
 *                         调用 processor.run() 阻塞执行，返回 exitcode
 *   - vx_bridge_destroy : 析构
 *
 * 注意：run() 在 QEMU BQL 持有期间同步执行。kernel 执行时间较短时
 * 可接受；若需要长时间执行应移到独立线程。
 */

// QEMU 基础设施头文件必须排在最前面
extern "C" {
#include "qemu/osdep.h"
#include "qemu/log.h"
}

// 然后是标准库
#include <cstdarg>
#include <cstdio>
#include <cstring>

// 在包含 SimX 之前，把 QEMU 冲突的宏临时取消
//    这里用 push/pop 保存现场，防止 SimX 里还有其它冲突宏（如 MAX, ARRAY_SIZE 等）
#pragma push_macro("MIN")
#undef MIN

// 上游 SimX / 桥接头文件
#include "vortex_bridge.h"
#include "arch.h"
#include "processor.h"
#include "simx_log.h"
#include "mem.h"
#include "VX_types.h"

// 恢复 QEMU 的 MIN（如果后续代码还需要 QEMU 的宏）
#pragma pop_macro("MIN")

#pragma GCC diagnostic ignored "-Wsuggest-attribute=format"

using namespace vortex;

/*
 * ---------------------------------------------------------------------------
 * QEMU 日志适配器：把 SimX 的回调接到 qemu_log
 * ---------------------------------------------------------------------------
 */

static void qemu_log_adapter(const char *fmt, va_list ap)
{
    char *msg = g_strdup_vprintf(fmt, ap);
    qemu_log("%s", msg);
    g_free(msg);
}

static bool simx_log_registered = false;

static void simx_log_register(void)
{
    if (!simx_log_registered) {
        simx_set_log_func(qemu_log_adapter);
        simx_log_registered = true;
        SIMX_LOG("[bridge] log callback registered to qemu_log");
    }
}

/*
 * ---------------------------------------------------------------------------
 * VRAM 布局常量
 * ---------------------------------------------------------------------------
 */

static const uint64_t VRAM_BASE      = 0;
static const uint64_t SIMX_RAM_SIZE  = 64 * 1024 * 1024;  // 64 MB，匹配 GPGPU VRAM

/*
 * ---------------------------------------------------------------------------
 * VxBridgeHandle
 * ---------------------------------------------------------------------------
 */

struct VxBridgeHandle {
    Arch       arch;
    RAM        ram;
    Processor  proc;

    VxBridgeHandle(uint32_t num_cores, uint32_t num_warps, uint32_t num_threads)
        : arch(num_threads, num_warps, 1)
        , ram(VRAM_BASE, SIMX_RAM_SIZE)
        , proc(arch)
    {
        SIMX_LOG("[bridge] VxBridgeHandle constructed: "
                 "cores=%u, warps=%u, threads=%u, ram_size=%lu",
                 num_cores, num_warps, num_threads, SIMX_RAM_SIZE);
        proc.attach_ram(&ram);
        proc.dcr_write(VX_DCR_BASE_MPM_CLASS, 0);
    }
};

/*
 * ---------------------------------------------------------------------------
 * 对外 C 接口
 * ---------------------------------------------------------------------------
 */

extern "C" VxBridgeHandle *
vx_bridge_create(uint32_t num_cores, uint32_t num_warps, uint32_t num_threads)
{
    simx_log_register();
    SIMX_LOG("[bridge] vx_bridge_create: cores=%u, warps=%u, threads=%u",
             num_cores, num_warps, num_threads);
    return new VxBridgeHandle(num_cores, num_warps, num_threads);
}

extern "C" void
vx_bridge_destroy(VxBridgeHandle *h)
{
    SIMX_LOG("[bridge] vx_bridge_destroy: handle=%p", h);
    delete h;
}

// 函数签名改为接收 block_dim[3]
extern "C" int
vx_bridge_run(VxBridgeHandle *h,
              const uint8_t  *vram,
              uint64_t        vram_size,
              uint64_t        kernel_addr,
              const uint32_t  grid_dim[3],
              const uint32_t  block_dim[3])
{
    SIMX_LOG("[bridge] === vx_bridge_run entry ===");
    SIMX_LOG("[bridge]   handle=%p, vram=%p, vram_size=%lu, kernel_addr=0x%lx, "
             "block_dim=(%u,%u,%u)",
             h, vram, vram_size, kernel_addr,
             block_dim[0], block_dim[1], block_dim[2]);

    /* 写入 VRAM 到 SimX RAM */
    h->ram.write(vram, VRAM_BASE, vram_size);
    SIMX_LOG("[bridge] RAM write done: %lu bytes at base 0x%lx",
             vram_size, VRAM_BASE);

    /* 设置 startup 地址 */
    uint64_t startup = VRAM_BASE + kernel_addr;
    h->proc.dcr_write(VX_DCR_BASE_STARTUP_ADDR0, (uint32_t)(startup & 0xffffffffULL));
    h->proc.dcr_write(VX_DCR_BASE_STARTUP_ADDR1, (uint32_t)(startup >> 32));
    SIMX_LOG("[bridge] startup addr=0x%lx", startup);

    /* 写入 block 维度 */
    h->proc.dcr_write(VX_DCR_BLOCK_DIM_X, block_dim[0]);
    h->proc.dcr_write(VX_DCR_BLOCK_DIM_Y, block_dim[1]);
    h->proc.dcr_write(VX_DCR_BLOCK_DIM_Z, block_dim[2]);

    /* 写入 grid 维度（为未来多 block 调度预留） */
    h->proc.dcr_write(VX_DCR_GRID_DIM_X, grid_dim[0]);
    h->proc.dcr_write(VX_DCR_GRID_DIM_Y, grid_dim[1]);
    h->proc.dcr_write(VX_DCR_GRID_DIM_Z, grid_dim[2]);

    SIMX_LOG("[bridge] grid_dim=(%u,%u,%u) block_dim=(%u,%u,%u)",
             grid_dim[0], grid_dim[1], grid_dim[2],
             block_dim[0], block_dim[1], block_dim[2]);

    /* 执行 */
    int exitcode = h->proc.run();
    SIMX_LOG("[bridge] Processor::run() returned exitcode=%d", exitcode);

    /* 读回 VRAM */
    h->ram.read(const_cast<uint8_t *>(vram), VRAM_BASE, vram_size);
    SIMX_LOG("[bridge] RAM read back done: %lu bytes", vram_size);

    /* 重建 Processor，保证每次执行的独立性 */
    h->proc.~Processor();
    new (&h->proc) Processor(h->arch);
    h->proc.attach_ram(&h->ram);
    h->proc.dcr_write(VX_DCR_BASE_MPM_CLASS, 0);
    SIMX_LOG("[bridge] Processor rebuilt for next run");

    SIMX_LOG("[bridge] === vx_bridge_run exit (ret=%d) ===", exitcode);
    return exitcode;
}