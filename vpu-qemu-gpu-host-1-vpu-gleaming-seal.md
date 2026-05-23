# VPU 独立进程化 & QEMU 管道化改造计划

## Context

当前 GPU 模拟器全部内嵌在 QEMU 进程中：PCI 设备模型、寄存器、VRAM、RISC-V SIMT 指令解释器都在同一个地址空间。这导致：
- VPU 核心与 QEMU 框架深度耦合（`GPGPUState*` 全局传递、`qemu/osdep.h` 强制依赖）
- 无法独立部署、调试、升级 VPU
- 日志系统缺失（`gpgpu_log.h/c` 不存在），观测性不足
- 现有 `ring.c` 环形缓冲区未接入任何模块

**目标**：QEMU 退化为薄管道层（仅负责 PCI/virtio 协议面 + 中断投递），VPU 成为独立的 host 进程，承载全部 GPU 模拟逻辑。

## 架构总览

```
┌─ Guest ──────────────────────────────────────────────────┐
│  userspace: gpgpu_runtime  (mmap VRAM, ioctl dispatch)   │
│  kernel:    gpgpu.ko       (PCI driver, BAR mmap/ioctl)  │
├──────────────────────────────────────────────────────────┤
│  QEMU (thin frontend)                                    │
│  ┌───────────────────┐  ┌──────────────────────────────┐ │
│  │ PCI/virtio device │  │ IRQ delivery (MSI-X/MSI/INT) │ │
│  │ - BAR 暴露        │  │ - kernel done → guest IRQ    │ │
│  │ - 共享内存映射     │  │ - error → guest IRQ         │ │
│  │ - doorbell 转发    │  │                              │ │
│  └──────┬────────────┘  └──────────┬───────────────────┘ │
│         │ eventfd/shared-mem       │ eventfd              │
├─────────┼──────────────────────────┼──────────────────────┤
│  VPU Process (host)                │                      │
│  ┌──────┴──────────────────────────┴───────────────────┐ │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │ │
│  │  │ Register │ │  VRAM    │ │ Command Queue      │  │ │
│  │  │ File     │ │ Manager  │ │ Processor          │  │ │
│  │  └──────────┘ └──────────┘ └─────────┬──────────┘  │ │
│  │  ┌───────────────────────────────────┴──────────┐   │ │
│  │  │ RISC-V SIMT Core (gpgpu_core)                │   │ │
│  │  │ - Warp scheduler / instruction interpreter   │   │ │
│  │  └──────────────────────┬───────────────────────┘   │ │
│  │  ┌──────────────────────┴───────────────────────┐   │ │
│  │  │ Observability (host process memory)          │   │ │
│  │  │ - Fast ring (4MB): level=0x00 指令 trace     │   │ │
│  │  │ - Slow ring (64KB): level=0x01 控制事件      │   │ │
│  │  └──────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  Observability Platform (host)                           │
│  - 接入 VPU 进程内存中的快/慢环                           │
│  - 解析二进制协议 → JSON / 可视化 / 性能分析              │
└──────────────────────────────────────────────────────────┘
```

## 设计决策

### 环形缓冲区：主机进程内存，不接共享内存

快慢环缓冲区在 VPU 进程的堆上分配（普通 `malloc`），仅供观测平台消费。观测平台通过 VPU 暴露的接口（如 debugfs、Unix socket、或直接作为线程嵌入 VPU）读取数据，不需要 `shm_open`/`mmap` 跨进程共享。

### 二进制协议层级（level）

协议头 4 位为层级（level），观测平台接收多个层级的信息。VPU 只占用两个层级：

| level | 用途 | 环 | 定义脚本 | 输出头文件 |
|-------|------|-----|----------|------------|
| `0x00` | 指令执行 trace | Fast ring | `gen_inst.py` | `pt_inst.h` |
| `0x01` | VPU 控制/状态事件 | Slow ring | `gen_event.py` | `pt_event.h` |

编码格式：`level[31:28] | nargs[27:24] | opcode[23:16] | flags[0]`，后跟 nargs×4B 操作数。

### Python 脚本统一管理

**目录结构**:
```
vpu/proto/
  main.py          # 编排入口，调用所有 gen_*.py
  gen_inst.py      # 生成 pt_inst.h (level=0x00, 95条RISC-V指令编码)
  gen_event.py     # 生成 pt_event.h (level=0x01, 控制事件编码)
  pt_inst.h        # 自动生成 (已有)
  pt_event.h       # 自动生成 (新建)
```

每个 `gen_*.py` 独立管理一种宏定义（一个 level），`main.py` 统一编排：
```python
# main.py — 协议宏生成入口
import gen_inst
import gen_event

gen_inst.generate("pt_inst.h")
gen_event.generate("pt_event.h")
```

### 不做 gpgpu_log 系统

不创建 `vpu/log/log.c` / `log.h`，不保留 `GPGPU_DEV`/`GPGPU_CORE`/`GPGPU_ERR` 等文本日志宏。**所有**日志输出统一写为二进制协议数据，进入快环或慢环：

- 指令 trace → `GPGPU_INST_BIN()`（复用已有 `INST_*` 宏，改写入 fast ring）
- 控制事件 → 新增 `EVENT_*` 宏（由 `pt_event.h` 生成），通过 `VPU_EVENT()` 写入 slow ring

---

## Phase 1: VPU 可观测系统（二进制协议 + 快慢环缓冲区）

### 1.1 改造 ring.c → 提供独立 API 头文件

**现有**: [vpu/ring/ring.c](hw/gpgpu/vpu/ring/ring.c) — 锁-free SPSC 环形缓冲区，4KB 固定大小，`_Atomic` 读写指针。功能完备但无头文件，未被任何模块引用。

**改造**:
- 新建 `vpu/ring/ring.h`，导出 API 声明
- 保持现有实现不变（进程内堆内存，`uint8_t buf[SIZE]` 嵌入 struct）
- 新增 `ring_buf_create(size_t size)` — 动态分配可配置大小的环

### 1.2 重组 gen.py + 新建 gen_event.py

**现有脚本**: [vpu/proto/gen.py](hw/gpgpu/vpu/proto/gen.py) — 生成 `pt_inst.h`，level=0x00，95 条 RISC-V 指令编码

**改造**:
- `gen.py` → `gen_inst.py`（重命名，内容不变，LEVEL=0x00）
- 新建 `gen_event.py` — 生成 `pt_event.h`，LEVEL=0x01，控制事件编码：
```
level=0x01 opcode:
  0x01 = REG_WRITE     — (offset, value)
  0x02 = REG_READ      — (offset, value)
  0x03 = DMA_START     — (src_lo, src_hi, dst_lo, dst_hi, size)
  0x04 = DMA_COMPLETE  — (status)
  0x05 = KERNEL_DISPATCH — (kernel_addr, grid_x, grid_y, grid_z, block_x, block_y, block_z)
  0x06 = KERNEL_COMPLETE — (status)
  0x07 = IRQ_FIRE      — (irq_type, vector)
  0x08 = ERROR_EVENT   — (error_code, detail)
  0x09 = STATE_CHANGE  — (state_old, state_new)
```
- 新建 `main.py` — 编排入口，顺序调用 `gen_inst.generate()` 和 `gen_event.generate()`

### 1.3 添加 make gen 目标

**文件**: `vpu/Makefile`

```makefile
gen:
	python3 proto/main.py
```

`make gen` 一键生成所有 `pt_*.h` 头文件（`pt_inst.h` + `pt_event.h`），源码中通过 `#include "proto/pt_inst.h"` / `#include "proto/pt_event.h"` 引用。

### 1.4 改造 proto.c → fast ring 替代文件

**文件**: [vpu/core/proto.c](hw/gpgpu/vpu/core/proto.c)

当前 `gpgpu_inst_trace_bin()` 写入 `inst_trace.bin` 文件（`FILE*` + `fwrite`）。改为写入 fast ring buffer：

- 外部传入 `ring_buf *fast_ring` 指针（通过全局或参数）
- 将 header + nargs×operand 打包为连续 buffer，调用 `log_write()` 写入 fast ring
- 移除 `fopen`/`fwrite`/`fclose` 文件操作

### 1.5 新增 VPU_EVENT 宏 → slow ring

**修改文件**: `vpu/core/proto.h` — 增加 `VPU_EVENT()` 宏，使用 `pt_event.h` 中的 `EVENT_*` 编码

```c
#include "proto/pt_event.h"

// 控制事件写入 slow ring，event_code 来自 pt_event.h (EVENT_REG_WRITE, etc.)
#define VPU_EVENT(ring, event_code, ...) \
    vpu_event_write(ring, event_code, ##__VA_ARGS__)
```

调用方示例：
```c
VPU_EVENT(s->slow_ring, EVENT_REG_WRITE, offset, value);
VPU_EVENT(s->slow_ring, EVENT_KERNEL_COMPLETE, status);
```

### 1.6 清理 QEMU 依赖

VPU core 所有文件中：
- 移除 `#include "qemu/osdep.h"`、`"qemu/log.h"`、`"gpgpu_log.h"`
- 移除 `qemu_log_mask()` 调用
- 替换为 `VPU_EVENT(slow_ring, ERROR_EVENT, ...)` 或直接删除（非关键 info 日志）

**受影响的文件**:
- [gpgpu_core.c:10-11,15](hw/gpgpu/vpu/core/gpgpu_core.c) — 移除 qemu includes + gpgpu_log.h
- [memory.c:10-12](hw/gpgpu/vpu/core/memory.c) — 替换 `qemu_log_mask` 为 `VPU_EVENT`
- [proto.c:5](hw/gpgpu/vpu/core/proto.c) — 移除 `qemu/osdep.h`

### Phase 1 验证

```
cd vpu && make          # 编译 libvpu.a
nm libvpu.a | grep -i qemu  # 无 QEMU 符号
```

---

## Phase 2: 解耦 VPU 和 QEMU

### 2.1 提炼 VPU 独立状态结构

**新文件**: `vpu/state.h` — VPU 进程内状态（纯 C11，零 QEMU 依赖）

```c
typedef struct {
    // 设备配置
    uint32_t num_cus;
    uint32_t warps_per_cu;
    uint32_t warp_size;
    uint64_t vram_size;

    // VRAM (指向共享内存或本地分配)
    uint8_t *vram_ptr;

    // 寄存器文件
    uint32_t global_ctrl;
    uint32_t global_status;
    uint32_t error_status;
    uint32_t irq_enable;
    uint32_t irq_status;
    GPGPUKernelParams kernel;
    GPGPUDMAState dma;
    GPGPUSIMTContext simt;

    // 环形缓冲区 (进程内堆内存)
    ring_buf *fast_ring;   // level=0x00 指令 trace, 4MB
    ring_buf *slow_ring;   // level=0x01 控制事件, 64KB
} VPUState;
```

`GPGPUKernelParams`, `GPGPUDMAState`, `GPGPUSIMTContext` 从 [gpgpu.h](hw/gpgpu/gpgpu.h) 移出，放入 `vpu/state.h`（去掉 QEMU 类型依赖）。

### 2.2 改造核心解释器签名

所有接受 `GPGPUState*` 的函数改为接受 `VPUState*`：

- `gpgpu_core_exec_kernel(VPUState*)`
- `gpu_read(VPUState*, uint32_t addr, int len)`
- `gpu_write(VPUState*, uint32_t addr, int len, uint32_t data)`

### 2.3 VPU 独立构建系统

**重写**: `vpu/Makefile`（替代 0 字节占位文件）

```makefile
# VPU standalone build (host)
TARGET  = libvpu.a
SRCS    = core/gpgpu_core.c core/memory.c core/proto.c core/lpfp.c ring/ring.c
OBJS    = $(SRCS:.c=.o)
CFLAGS  = -Wall -Wextra -O2 -g -std=c11 -I. -Icore -Iring -Iproto

$(TARGET): $(OBJS)
	ar rcs $@ $^

gen:
	python3 proto/main.py

clean:
	rm -f $(OBJS) $(TARGET)
```

- `make` — 编译 `libvpu.a`
- `make gen` — 调用 `proto/main.py` 生成所有 `pt_*.h`

### 2.4 更新 QEMU meson.build

[meson.build](hw/gpgpu/meson.build)：移除 VPU 源文件，只保留 QEMU 前端：

```meson
system_ss.add(when: 'CONFIG_GPGPU', if_true: files(
    'gpgpu.c',
))
```

### Phase 2 验证

```
make -C vpu              # libvpu.a 编译成功
make -C guest            # guest 代码编译不受影响
```

---

## Phase 3: QEMU-VPU 交互协议

### 3.1 通信通道设计

```
通道           方向          机制              用途
──────────────────────────────────────────────────────────
VRAM           双向          共享内存 (shm)     GPU 显存，guest 通过 QEMU BAR2 可见
CTRL           双向          共享内存 (shm)     寄存器读写转发
Doorbell       Guest→VPU     eventfd           内核分发通知
Completion     VPU→QEMU      eventfd           内核完成 → QEMU 触发 IRQ
Error          VPU→QEMU      eventfd           错误 → QEMU 触发 IRQ
Fast Ring      VPU 内部      进程堆内存         指令 trace → 观测平台
Slow Ring      VPU 内部      进程堆内存         控制事件 → 观测平台
```

### 3.2 前后端共享接口定义

**新文件**: `vpu/iface.h` — QEMU 和 VPU 共识的协议常量

```c
// 共享内存名称
#define VPU_SHM_VRAM_NAME   "/vpu_vram"
#define VPU_SHM_CTRL_NAME   "/vpu_ctrl"

// eventfd 名称 (通过环境变量或命令行传递 fd)
#define VPU_ENV_DOORBELL_FD  "VPU_DOORBELL_FD"
#define VPU_ENV_COMPLETE_FD  "VPU_COMPLETE_FD"
#define VPU_ENV_ERROR_FD     "VPU_ERROR_FD"

// 控制通道命令
#define VPU_CMD_NOP          0
#define VPU_CMD_REG_WRITE    1   // data[0]=offset, data[1]=value
#define VPU_CMD_REG_READ     2   // data[0]=offset → VPU 填充 data[1]=value
#define VPU_CMD_DISPATCH     3   // 启动内核执行
#define VPU_CMD_RESET        4   // 软复位
```

### 3.3 QEMU 前端职责（缩减后）

[gpgpu.c](hw/gpgpu/gpgpu.c) 只保留：

1. **PCI 设备注册 & BAR 暴露**
   - BAR0（ctrl MMIO）：每次 guest 读写时，转换为 `VPU_CMD_REG_READ/WRITE` 写入 CTRL 共享内存通道
   - BAR2（VRAM）：`memory_region_init_ram_fd()` 映射 VPU 的 VRAM 共享内存，guest 直接 mmap
   - BAR4（doorbell）：写时通过 eventfd 通知 VPU

2. **中断投递**
   - 监听 completion eventfd（通过 `qemu_set_fd_handler` 或 iohandler）
   - 收到通知后触发 MSI-X/MSI/INTx 中断

3. **设备生命周期**
   - realize: 创建共享内存、启动 VPU 子进程、传递 fd
   - exit: 关闭 fd、终止 VPU 子进程
   - reset: 发送 `VPU_CMD_RESET`

### 3.4 VPU 主循环

**新文件**: `vpu/main.c` — VPU 进程入口

```c
int main(int argc, char **argv) {
    VPUState state;
    vpu_state_init(&state, argc, argv);  // 解析 env 获取 eventfd
    observability_init(&state);          // 分配 fast/slow ring

    while (!state.should_exit) {
        eventfd_read(doorbell_fd, &val); // 阻塞等待 QEMU doorbell
        VPUCtrlCmd cmd;
        while (vpu_ctrl_read_cmd(&state, &cmd)) {
            switch (cmd.op) {
            case VPU_CMD_REG_WRITE:
                vpu_reg_write(&state, cmd.data[0], cmd.data[1]);
                VPU_EVENT(state.slow_ring, REG_WRITE, cmd.data[0], cmd.data[1]);
                break;
            case VPU_CMD_REG_READ:
                cmd.data[1] = vpu_reg_read(&state, cmd.data[0]);
                vpu_ctrl_write_response(&state, &cmd);
                break;
            case VPU_CMD_DISPATCH:
                VPU_EVENT(state.slow_ring, KERNEL_DISPATCH, ...);
                gpgpu_core_exec_kernel(&state);
                VPU_EVENT(state.slow_ring, KERNEL_COMPLETE, state.error_status);
                eventfd_write(complete_fd, 1); // 通知 QEMU 投递 IRQ
                break;
            case VPU_CMD_RESET:
                vpu_state_reset(&state);
                break;
            }
        }
    }
}
```

### 3.5 Guest 侧影响

**Zero change** — guest 驱动和运行时完全不感知后端变化：
- `/dev/gpgpuN` 照常打开
- mmap BAR2 照常工作（QEMU 将 VPU 共享内存映射为 BAR2）
- ioctl 照常工作（QEMU 转发 MMIO 到 VPU）

---

## 实施顺序

Phase 1 → 2 → 3 顺序实施（Phase 1 是 Phase 2/3 的基础依赖）。

### Phase 1 子任务
1. 创建 `ring.h` 头文件，导出 ring.c API
2. 定义 `ctrl_proto.h`（level=0x01 事件编码）
3. 改造 `proto.c` → 写入 fast ring 替代文件
4. 新增 `VPU_EVENT()` 宏 → 写入 slow ring
5. 清理 VPU 所有 QEMU 依赖（`qemu/osdep.h`, `qemu/log.h`, `qemu_log_mask`, `gpgpu_log.h`）
6. 验证：`make -C vpu` 编译通过，`nm libvpu.a | grep -i qemu` 无输出

### Phase 2 子任务
1. 创建 `vpu/state.h`（VPUState，从 gpgpu.h 提炼）
2. 改造函数签名 `GPGPUState*` → `VPUState*`
3. 编写 `vpu/Makefile`
4. 更新 `hw/gpgpu/meson.build`
5. 验证：VPU 独立编译成功

### Phase 3 子任务
1. 创建 `vpu/iface.h`（前后端协议常量）
2. 实现 QEMU 侧 `gpgpu.c` 缩减（MMIO 转发 + 中断投递）
3. 实现 `vpu/main.c`（VPU 主循环）
4. 实现共享内存通道（VRAM shm + CTRL shm）
5. 实现 eventfd 通知链（doorbell, completion, error）
6. 端到端测试：`make test-kernel-vector_add` 通过

---

## 关键文件清单

| 文件 | 操作 |
|------|------|
| `vpu/ring/ring.c` | 改造（增加动态大小分配） |
| `vpu/ring/ring.h` | **新建** |
| `vpu/proto/main.py` | **新建**（编排入口，调用所有 gen_*.py） |
| `vpu/proto/gen_inst.py` | 重命名自 `gen.py`（level=0x00） |
| `vpu/proto/gen_event.py` | **新建**（level=0x01 控制事件） |
| `vpu/proto/pt_inst.h` | 自动生成（已有） |
| `vpu/proto/pt_event.h` | 自动生成（**新建**） |
| `vpu/core/proto.c` | 改造（文件→fast ring） |
| `vpu/core/proto.h` | 改造（增加 VPU_EVENT 宏） |
| `vpu/core/gpgpu_core.c` | 改造（移除QEMU依赖、`GPGPUState*`→`VPUState*`、文本日志→二进制事件） |
| `vpu/core/gpgpu_core.h` | 改造 |
| `vpu/core/memory.c` | 改造（`qemu_log_mask`→`VPU_EVENT`） |
| `vpu/core/memory.h` | 改造 |
| `vpu/state.h` | **新建**（VPU 独立状态） |
| `vpu/iface.h` | **新建**（前后端共享协议常量） |
| `vpu/main.c` | **新建**（VPU 进程入口） |
| `vpu/Makefile` | 重写（`make` + `make gen`） |
| `hw/gpgpu/gpgpu.c` | 大幅简化（QEMU 薄管道） |
| `hw/gpgpu/gpgpu.h` | 简化（移除 GPU 内部状态，移到 vpu/state.h） |
| `hw/gpgpu/meson.build` | 更新（移除 VPU 源文件） |

## 不做的事

- ~~`vpu/log/log.c` / `vpu/log/log.h`~~ — 不创建日志系统，统一写环形缓冲区
- ~~环形缓冲区共享内存~~ — 环在 VPU 进程堆内，观测平台通过进程内接口读取
- ~~`GPGPU_DEV`/`GPGPU_CORE`/`GPGPU_ERR` 文本宏~~ — 全部替换为二进制 `VPU_EVENT()`

## 验证方式

1. **Phase 1**: `nm libvpu.a | grep -i qemu` 返回空
2. **Phase 2**: `make -C vpu && make -C guest` 均编译通过
3. **Phase 3**: `make test-kernel-vector_add` 通过，guest 行为与改造前一致
4. **观测验证**: 观测平台能解析 fast ring（level=0x00 指令 trace）和 slow ring（level=0x01 控制事件）
