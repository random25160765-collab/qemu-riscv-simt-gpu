# VPU 二进制观测协议

## 记录格式

```
每条记录 = header(4B) + operand_0(4B) + operand_1(4B) + ... + operand_{nargs-1}(4B)
```

### Header 编码 (32-bit)

```
 31    28 27    24 23    16 15       1 0
+--------+--------+--------+---------+-+
| level  | nargs  | opcode | reserved|f|
+--------+--------+--------+---------+-+
```

| 域 | 位 | 说明 |
|----|-----|------|
| level | [31:28] | 层级，VPU 占用 `0x0` 和 `0x1` |
| nargs | [27:24] | 操作数个数 (0-15)，每个操作数 4B |
| opcode | [23:16] | 事件/指令编号 (0-255) |
| reserved | [15:1] | 保留，填 0 |
| f (flags) | [0] | level=0x0 时为 branch 标志；level=0x1 时保留 |

### 64-bit 值编码约定

所有 64-bit 值（地址等）拆为两个操作数 slot，**低 32 位在前，高 32 位在后**。

---

## level=0x0 — 指令执行 Trace (fast ring)

已定义在 `gen_inst.py` → `pt_inst.h`，95 条 RISC-V 指令，format 同上。

bit[0] 为 branch 标志：1=跳转指令，0=非跳转指令。

示例：
```
INST_JAL  = 0x03010001  // level=0, nargs=3, opcode=1,  branch=1
INST_ADD  = 0x031C0000  // level=0, nargs=3, opcode=28, branch=0
```

---

## level=0x1 — VPU 控制事件 (slow ring)

定义在 `gen_event.py` → `pt_event.h`。

### 事件列表

| opcode | 宏名 | nargs | 操作数 |
|--------|------|-------|--------|
| 0x01 | `REG_WRITE` | 2 | `offset`, `value` |
| 0x02 | `REG_READ` | 2 | `offset`, `value` |
| 0x03 | `DMA_START` | 5 | `src_lo`, `src_hi`, `dst_lo`, `dst_hi`, `size` |
| 0x04 | `DMA_COMPLETE` | 1 | `status` |
| 0x05 | `KERNEL_DISPATCH` | 11 | `kernel_addr_lo`, `kernel_addr_hi`, `args_addr_lo`, `args_addr_hi`, `grid_x`, `grid_y`, `grid_z`, `block_x`, `block_y`, `block_z`, `shared_mem` |
| 0x06 | `KERNEL_COMPLETE` | 1 | `status` |
| 0x07 | `IRQ_FIRE` | 2 | `irq_type`, `vector` |
| 0x08 | `ERROR_EVENT` | 2 | `error_code`, `detail` |
| 0x09 | `STATE_CHANGE` | 2 | `state_old`, `state_new` |

### 各事件说明

#### REG_WRITE / REG_READ
MMIO 寄存器写/读。`offset` 为寄存器地址，`value` 为 32-bit 值。

#### DMA_START
DMA 传输发起。`src`/`dst` 为 64-bit 地址，各拆为 lo/hi 两个 slot。`size` 为传输字节数。

#### DMA_COMPLETE
DMA 传输完成。`status` 值：0=成功，bit0=busy，bit1=complete，bit2=error。

#### KERNEL_DISPATCH
内核分发。`kernel_addr`/`args_addr` 为 64-bit VRAM 地址各拆为 lo/hi。`grid_*`/`block_*` 为三维维度，`shared_mem` 为共享内存字节数。

#### KERNEL_COMPLETE
内核执行完成。`status` 值同 GLOBAL_STATUS 寄存器。

#### IRQ_FIRE
中断触发。`irq_type`：0=KERNEL_DONE，1=DMA_DONE，2=ERROR。`vector` 为 MSI-X vector 编号，legacy INTx 时为 -1。

#### ERROR_EVENT
错误事件。`error_code` 按 ERROR_STATUS 寄存器位定义。`detail` 为额外上下文（如越界地址）。

#### STATE_CHANGE
设备状态变更。`state_old`/`state_new` 值：0=READY，1=BUSY，2=ERROR。

### 示例编码

```
EVENT_REG_WRITE        = 0x12010000  // level=1, nargs=2, opcode=0x01
EVENT_DMA_START        = 0x15030000  // level=1, nargs=5, opcode=0x03
EVENT_KERNEL_DISPATCH  = 0x1B050000  // level=1, nargs=11, opcode=0x05
EVENT_KERNEL_COMPLETE  = 0x11060000  // level=1, nargs=1, opcode=0x06
```
