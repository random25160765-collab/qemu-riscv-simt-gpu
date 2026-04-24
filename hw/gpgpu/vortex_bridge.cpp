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

#include "vortex_bridge.h"

#include "simx/arch.h"
#include "simx/processor.h"

/* common 层头文件（来自上游 vortex/sim/common/） */
#include "mem.h"
#include "VX_types.h"

#include <cstdio>
#include <cstring>

using namespace vortex;

/*
 * VRAM 在 SimX 地址空间的基地址。
 * QTest kernel 直接用小地址（0x0 = 代码, 0x1000 = 数据），
 * IO 区域只有 0x40~0x7f，不冲突，直接从 0 映射。
 */
static const uint64_t VRAM_BASE = 0;

struct VxBridgeHandle {
    Arch       arch;
    RAM        ram;
    Processor  proc;

    VxBridgeHandle(uint32_t num_cores, uint32_t num_warps, uint32_t num_threads)
        : arch(num_threads, num_warps, num_cores)
        , ram(0, MEM_PAGE_SIZE)
        , proc(arch)
    {
        proc.attach_ram(&ram);
        proc.dcr_write(VX_DCR_BASE_MPM_CLASS, 0);
    }
};

extern "C" VxBridgeHandle *
vx_bridge_create(uint32_t num_cores, uint32_t num_warps, uint32_t num_threads)
{
    return new VxBridgeHandle(num_cores, num_warps, num_threads);
}

extern "C" void
vx_bridge_destroy(VxBridgeHandle *h)
{
    delete h;
}

extern "C" int
vx_bridge_run(VxBridgeHandle *h,
              const uint8_t  *vram,
              uint64_t        vram_size,
              uint64_t        kernel_addr)
{
    /* 把整块 VRAM 写入 SimX RAM，映射到 VRAM_BASE 处 */
    h->ram.write(vram, VRAM_BASE, vram_size);

    /* startup_addr = VRAM_BASE + kernel 在 VRAM 内的偏移 */
    uint64_t startup = VRAM_BASE + kernel_addr;
    h->proc.dcr_write(VX_DCR_BASE_STARTUP_ADDR0, (uint32_t)(startup & 0xffffffffULL));
    h->proc.dcr_write(VX_DCR_BASE_STARTUP_ADDR1, (uint32_t)(startup >> 32));

    int exitcode = h->proc.run();

    /* run() 后 Processor 处于 halted 状态，原地重建以便下次使用 */
    h->proc.~Processor();
    new (&h->proc) Processor(h->arch);
    h->proc.attach_ram(&h->ram);
    h->proc.dcr_write(VX_DCR_BASE_MPM_CLASS, 0);

    return exitcode;
}
