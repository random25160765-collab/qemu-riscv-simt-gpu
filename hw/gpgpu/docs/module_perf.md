好的，我来把关于性能模块的讨论整理成一份结构化 Spec，可以直接交给 Agent 执行。

---

# RVSIM 性能测量框架规格说明书

## 一、概述

### 1.1 目标
为 RVSIM 建立**低扰动、分层、可扩展**的性能测量框架，支撑以下分析需求：
- 端到端 kernel 延迟测量
- 软件栈开销分解（Runtime → 驱动 → QEMU → SIMT 核）
- 长时间 profiling（热点分析、瓶颈定位）
- 硬件计数器暴露（IPC、停顿率、分支发散率）

### 1.2 设计原则
| 原则 | 说明 |
|:---|:---|
| **测量与输出分离** | 运行时只记录，不打印；事后批量导出 |
| **硬件计数器零开销** | SIMT 核计数器为寄存器递增，不额外消耗周期 |
| **Hash 解耦语义** | 设备层只知道 kernel hash，不解析算子名 |
| **编译期可选** | 所有测量代码由 `CONFIG_PERF` 宏控制，可完全抹除 |
| **校准测量自身开销** | 提供空 kernel 基准，修正探头效应 |

---

## 二、分层架构

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Runtime (C)                                        │
│ - 端到端计时 (clock_gettime)                                 │
│ - Kernel hash 生成与注册                                     │
│ - 报告汇总与反向映射                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Driver (Linux kernel)                              │
│ - ioctl 边界计时 (ktime_get_ns)                              │
│ - 性能计数器读出接口 (sysfs / ioctl)                          │
│ - 环形日志缓冲区                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: QEMU Device                                        │
│ - 命令处理计时 (qemu_clock_get_ns)                           │
│ - 长时间 Profiler (定时采样)                                  │
│ - 硬件计数器 MMIO 暴露                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: SIMT Core                                          │
│ - 周期级计数器 (cycles, insts, stalls, divergence)           │
│ - 零开销递增 (硬件风格寄存器)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、数据结构定义

### 3.1 通用 Hash 类型
```c
typedef uint32_t kernel_hash_t;   // FNV-1a 32-bit

// 特殊值
#define KERNEL_HASH_IDLE  0x00000000
```

### 3.2 分层计时记录

```c
// Layer 1: Runtime 记录
typedef struct runtime_perf_record {
    kernel_hash_t kernel_hash;
    uint64_t start_ns;      // clock_gettime(CLOCK_MONOTONIC)
    uint64_t end_ns;
    uint32_t grid[3];
    uint32_t block[3];
} runtime_perf_record_t;

// Layer 2: Driver 记录
typedef struct driver_perf_record {
    kernel_hash_t kernel_hash;
    uint64_t ioctl_entry_ns;    // ktime_get_ns()
    uint64_t ioctl_exit_ns;
    uint64_t mmio_submit_ns;
    uint64_t irq_received_ns;
} driver_perf_record_t;

// Layer 3: QEMU 设备记录
typedef struct qemu_perf_record {
    kernel_hash_t kernel_hash;
    uint64_t cmd_received_ns;   // qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL)
    uint64_t cmd_completed_ns;
    uint64_t dma_bytes;
} qemu_perf_record_t;

// Layer 4: SIMT 核心计数器
typedef struct simt_perf_counters {
    uint64_t cycles;            // 总执行周期
    uint64_t insts;             // 总执行指令数
    uint64_t fetch_stalls;      // 取指停顿周期
    uint64_t data_stalls;       // 访存停顿周期
    uint64_t branch_divergence; // 分支发散次数
    uint64_t warp_idle_cycles;  // warp 空闲周期
    uint32_t active_warps;      // 平均活跃 warp 数 (累加后平均)
} simt_perf_counters_t;
```

### 3.3 长时间 Profiler 数据结构

```c
#define PROF_MAX_KERNELS     64
#define PROF_SAMPLE_INTERVAL_MS  10

typedef struct profiler_sample_bucket {
    kernel_hash_t hash;
    uint64_t count;
} profiler_sample_bucket_t;

typedef struct qemu_profiler_state {
    QEMUTimer *timer;
    uint32_t current_kernel_hash;   // 0 = idle
    
    profiler_sample_bucket_t kernel_samples[PROF_MAX_KERNELS];
    int num_kernels;
    
    uint64_t dma_samples;
    uint64_t idle_samples;
    uint64_t total_samples;
    
    bool enabled;
} qemu_profiler_state_t;
```

### 3.4 聚合报告结构

```c
#define PERF_RING_SIZE  256

typedef struct gpgpu_profile_report {
    // 长时间 Profiler 结果
    uint32_t num_kernels;
    struct {
        kernel_hash_t hash;
        uint64_t sample_count;
    } kernel_hist[PROF_MAX_KERNELS];
    uint64_t dma_samples;
    uint64_t idle_samples;
    uint64_t total_samples;
    uint32_t sample_interval_ms;
    
    // 最近 N 次 kernel 的 SIMT 计数器快照
    uint32_t num_recent;
    struct {
        kernel_hash_t hash;
        simt_perf_counters_t counters;
    } recent_kernels[PERF_RING_SIZE];
    
    // 累计 SIMT 计数器
    simt_perf_counters_t accumulated;
} gpgpu_profile_report_t;
```

---

## 四、模块接口

### 4.1 Runtime 层 API

```c
// 初始化/销毁
void gpgpu_perf_init(void);
void gpgpu_perf_shutdown(void);

// Kernel 注册 (生成 hash)
kernel_hash_t gpgpu_perf_register_kernel(const char *name);
const char* gpgpu_perf_hash_to_name(kernel_hash_t hash);

// 计时 (内部由 launch/sync 自动调用，不暴露给用户)
void gpgpu_perf_record_launch_start(kernel_hash_t hash);
void gpgpu_perf_record_launch_end(kernel_hash_t hash);

// 获取报告
int gpgpu_perf_get_report(gpgpu_profile_report_t *report);

// 控制开关
void gpgpu_perf_enable(bool enabled);
void gpgpu_profiler_start(void);
void gpgpu_profiler_stop(void);

// 导出原始数据 (供离线分析)
int gpgpu_perf_dump_log(const char *filename);
```

### 4.2 驱动层接口

```c
// ioctl 命令
#define GPGPU_IOCTL_GET_PERF_REPORT  _IOR('G', 0x10, gpgpu_profile_report_t)
#define GPGPU_IOCTL_PERF_ENABLE      _IOW('G', 0x11, int)
#define GPGPU_IOCTL_PROFILER_START   _IO('G', 0x12)
#define GPGPU_IOCTL_PROFILER_STOP    _IO('G', 0x13)
#define GPGPU_IOCTL_RESET_COUNTERS   _IO('G', 0x14)

// sysfs 属性 (调试用)
// /sys/class/gpgpu/gpgpu0/perf/cycles
// /sys/class/gpgpu/gpgpu0/perf/insts
// /sys/class/gpgpu/gpgpu0/perf/ipc
// /sys/class/gpgpu/gpgpu0/perf/stall_rate
```

### 4.3 QEMU 设备 MMIO 寄存器

| 偏移 | 寄存器 | 访问 | 说明 |
|:---|:---|:---|:---|
| 0x100 | `PERF_KERNEL_HASH` | W | 当前执行的 kernel hash |
| 0x108 | `PERF_CTRL` | R/W | bit0: enable, bit1: profiler_start, bit2: profiler_stop |
| 0x110 | `PERF_CYCLES` | R | SIMT 周期计数器 |
| 0x118 | `PERF_INSTS` | R | SIMT 指令计数器 |
| 0x120 | `PERF_FETCH_STALLS` | R | 取指停顿 |
| 0x128 | `PERF_DATA_STALLS` | R | 访存停顿 |
| 0x130 | `PERF_BRANCH_DIV` | R | 分支发散次数 |
| 0x138 | `PERF_WARP_IDLE` | R | warp 空闲周期 |
| 0x140 | `PERF_REPORT_ADDR` | W | DMA 地址，用于批量写出报告 |
| 0x148 | `PERF_REPORT_SIZE` | R/W | 报告大小 |
| 0x150 | `PERF_REPORT_TRIGGER` | W | 写任意值触发 DMA 写出 |

### 4.4 SIMT 核心内部接口

```c
// 计数器递增 (内联，零开销)
static inline void simt_perf_inc_cycle(GPGPUCore *core) {
#ifdef CONFIG_PERF
    core->perf.cycles++;
#endif
}

static inline void simt_perf_inc_inst(GPGPUCore *core) {
#ifdef CONFIG_PERF
    core->perf.insts++;
#endif
}

static inline void simt_perf_inc_fetch_stall(GPGPUCore *core) {
#ifdef CONFIG_PERF
    core->perf.fetch_stalls++;
#endif
}

// 批量读取
void simt_perf_snapshot(GPGPUCore *core, simt_perf_counters_t *dst);
void simt_perf_reset(GPGPUCore *core);
```

---

## 五、关键流程

### 5.1 单次 Kernel 执行测量流程

```
1. Runtime: gpgpu_perf_record_launch_start(hash)
   └── 记录 start_ns

2. Runtime → Driver: ioctl(GPGPU_LAUNCH_KERNEL)
   └── Driver: 记录 ioctl_entry_ns

3. Driver → QEMU: MMIO 写 DOORBELL
   └── Driver: 记录 mmio_submit_ns
   └── QEMU: 记录 cmd_received_ns

4. QEMU: SIMT 核心执行
   └── 周期计数器自动递增

5. QEMU: 执行完成，触发中断
   └── QEMU: 记录 cmd_completed_ns
   └── 可选: 抓取 SIMT 计数器快照存入环形缓冲

6. Driver: 中断处理
   └── 记录 irq_received_ns

7. Driver → Runtime: ioctl 返回
   └── Driver: 记录 ioctl_exit_ns

8. Runtime: gpgpu_perf_record_launch_end()
   └── 记录 end_ns
   └── 将完整记录写入环形缓冲
```

### 5.2 长时间 Profiler 采样流程

```
1. Runtime: gpgpu_profiler_start()
   └── Driver: ioctl(GPGPU_PROFILER_START)

2. QEMU: 启动定时器 (每 10ms)
   └── timer_cb() 读取 current_kernel_hash
   └── kernel_samples[hash]++ 或 idle_samples++

3. (系统运行，处理多个 kernel)

4. Runtime: gpgpu_profiler_stop()
   └── Driver: ioctl(GPGPU_PROFILER_STOP)

5. Runtime: gpgpu_perf_get_report()
   └── Driver: ioctl(GPGPU_GET_PERF_REPORT)
   └── QEMU: 聚合 profiler_state + 环形缓冲 → 报告
   └── Runtime: hash → name 反向映射，打印结果
```

### 5.3 报告导出流程 (离线分析)

```
1. Runtime: gpgpu_perf_dump_log("profile.bin")
   └── 将环形缓冲 + 映射表序列化为二进制

2. 运行后: python parse_profile.py profile.bin
   └── 读取 hash→name 映射
   └── 解析各层记录
   └── 输出:
       - 各 kernel 累计耗时
       - 时间线可视化
       - 分层开销占比
```

---

## 六、基准校准模块

### 6.1 空 Kernel 基准

```c
// 空 kernel 汇编
// empty.S
empty_kernel:
    ret

// Runtime 校准流程
void gpgpu_perf_calibrate(void) {
    // 测 1000 次空 kernel，取中位数
    uint64_t samples[1000];
    for (int i = 0; i < 1000; i++) {
        samples[i] = measure_empty_kernel_overhead();
    }
    uint64_t overhead_ns = median(samples, 1000);
    
    // 存为全局校准值
    g_calibration.software_overhead_ns = overhead_ns;
}
```

### 6.2 计时函数自身开销

```c
uint64_t measure_clock_gettime_overhead(void) {
    struct timespec t1, t2;
    uint64_t sum = 0;
    for (int i = 0; i < 100000; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        sum += timespec_diff_ns(&t1, &t2);
    }
    return sum / 100000;
}
```

---

## 七、编译选项

```makefile
# Makefile / Kconfig
CONFIG_PERF=y                # 总开关
CONFIG_PERF_RUNTIME=y        # Runtime 层计时
CONFIG_PERF_DRIVER=y         # 驱动层计时
CONFIG_PERF_QEMU=y           # QEMU 设备计时
CONFIG_PERF_SIMT=y           # SIMT 核心计数器
CONFIG_PERF_PROFILER=y       # 长时间采样器
CONFIG_PERF_RING_SIZE=256    # 环形缓冲大小
```

---

## 八、输出示例

### 8.1 单次 Kernel 详细报告

```
========== Kernel Performance Report ==========
Kernel:      conv2d (hash: 0x7f3a2b1c)
Grid:        16x16x1
Block:       8x8x1

Latency breakdown:
  End-to-end:              1245.3 us  (100.0%)
  ├── Runtime overhead:     112.5 us  (  9.0%)
  ├── Driver (ioctl):        78.2 us  (  6.3%)
  ├── QEMU MMIO/DMA:        234.8 us  ( 18.9%)
  └── SIMT execution:       819.8 us  ( 65.8%)
      ├── Active cycles:    491.9 us  ( 60% of SIMT)
      └── Stall cycles:     327.9 us  ( 40% of SIMT)
          ├── Fetch stall:  196.7 us
          ├── Data stall:    98.4 us
          └── Warp idle:     32.8 us

SIMT counters:
  Cycles:          2,459,400
  Instructions:      614,850
  IPC:                   0.25
  Branch divergence:     142 events
```

### 8.2 长时间 Profiler 报告

```
========== Profile Report (10ms sampling, 1000 inferences, 48.2s) ==========

Kernel breakdown:
  conv2d        : 2145 samples (44.5%)  ██████████████████████████████████████████████
  matmul        :  856 samples (17.7%)  █████████████████████
  maxpool       :  423 samples ( 8.8%)  ██████████
  relu          :  134 samples ( 2.8%)  ███
  softmax       :  112 samples ( 2.3%)  ██
  (other)       :  577 samples (12.0%)  ████████████

System breakdown:
  DMA           :  356 samples ( 7.4%)
  Idle          :  576 samples (11.9%)

Top bottlenecks:
  1. conv2d - 44.5% of time, IPC=0.21 → 考虑向量化或分块
  2. Idle   - 11.9% of time → 检查 warp 调度器空转
  3. DMA    -  7.4% of time → 考虑双缓冲
```

---

## 九、交付物清单

| 文件 | 说明 |
|:---|:---|
| `runtime/gpgpu_perf.c` | Runtime 层性能模块 |
| `runtime/gpgpu_perf.h` | Runtime 层性能接口 |
| `driver/gpgpu_perf.c` | 驱动层性能模块 |
| `driver/gpgpu_perf.h` | 驱动层性能接口 |
| `hw/misc/gpgpu_perf.c` | QEMU 设备性能模块 |
| `hw/misc/gpgpu_perf.h` | QEMU 设备性能接口 |
| `hw/misc/gpgpu_core_perf.c` | SIMT 核心计数器实现 |
| `scripts/parse_profile.py` | 离线报告解析脚本 |
| `kernels/empty.S` | 空 kernel (基准校准用) |
| `tests/test_perf.c` | 性能框架单元测试 |
| `doc/perf_framework.md` | 使用文档 |

---

## 十、验收标准

| 编号 | 验收项 | 标准 |
|:---|:---|:---|
| AC-1 | 端到端计时 | 可测量单 kernel 延迟，误差 <5% |
| AC-2 | 分层开销分解 | 能区分 Runtime/Driver/QEMU/SIMT 耗时 |
| AC-3 | SIMT 计数器 | 正确暴露 IPC、停顿率、分支发散 |
| AC-4 | 长时间 Profiler | 运行 1000 次推理，采样报告误差 <2% |
| AC-5 | 零开销编译 | `CONFIG_PERF=n` 时，性能模块完全抹除 |
| AC-6 | 基准校准 | 空 kernel 开销已扣除，计时函数自身开销已知 |
| AC-7 | Hash 无冲突 | 64 个 kernel 以内无碰撞检测 |
| AC-8 | 离线分析 | Python 脚本可解析二进制日志并输出可读报告 |

---

此 Spec 可直接交给 Agent 执行，每个模块独立可测。如需调整粒度或补充细节，请提出。