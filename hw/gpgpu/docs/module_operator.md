## VRAM 完整布局规范

### 总览

| 地址范围 | 大小 | 用途 |
|:---|:---|:---|
| `0x000000 - 0x0FFFFF` | 1MB | Kernel 代码区 |
| `0x100000 - 0x1FFFFF` | 1MB | 输入数据区 A |
| `0x200000 - 0x2FFFFF` | 1MB | 输入数据区 B / 权重区 |
| `0x300000 - 0x3FFFFF` | 1MB | 输出数据区 C |
| `0x400000 - 0x7FFFFF` | 4MB | 临时缓冲区 / 中间结果 |

### 1. Kernel 代码区 (`0x000000`)

| 偏移 | 用途 |
|:---|:---|
| `0x0000` | Conv2D kernel |
| `0x1000` | ReLU kernel |
| `0x2000` | MaxPool kernel |
| `0x3000` | MatMul kernel |
| `0x4000` | Softmax kernel |
| `0x5000` | VectorAdd kernel |
| `0x6000` | BiasAdd kernel |
| `0x7000` | MaxPoolMulti kernel |
| ... | ... |

每个 kernel 最大 4KB，足够容纳当前所有算子。

---

### 2. 数据区 (`0x100000 - 0x3FFFFF`)

| 地址 | 约定 |
|:---|:---|
| `0x100000` | **主输入**：所有算子的第一输入 |
| `0x200000` | **次输入/权重**：第二输入或权重矩阵 |
| `0x300000` | **主输出**：所有算子的输出 |

**调用约定**：算子执行前，输入必须放在 `0x100000`（和 `0x200000`）；执行后，结果在 `0x300000`。

---

### 3. 临时缓冲区 (`0x400000 - 0x7FFFFF`)

用于存放中间结果。组装 CNN 时，层与层之间需要手动搬移数据。

**示例**：
```c
// Conv1 输出在 0x300000，需要作为 ReLU 的输入
// ReLU 是原地操作，不需要搬移

// ReLU 输出还在 0x300000，需要作为 MaxPool 输入
memcpy(vram + 0x100000, vram + 0x300000, size);
run_maxpool(...);
// MaxPool 输出在 0x200000，需要搬到 0x100000 给下一层
memcpy(vram + 0x100000, vram + 0x200000, size);
```

---

### 4. 算子参数读取

/*
 * ============================================================================
 * CTRL 设备地址定义 (GPU 核心视角)
 * ============================================================================
 * GPU 核心通过访问这些地址来获取自己的线程 ID
 */
#define GPGPU_CORE_CTRL_BASE        0x80000000  /* CTRL 基地址 */
#define GPGPU_CORE_CTRL_THREAD_ID_X (GPGPU_CORE_CTRL_BASE + 0x00)
#define GPGPU_CORE_CTRL_THREAD_ID_Y (GPGPU_CORE_CTRL_BASE + 0x04)
#define GPGPU_CORE_CTRL_THREAD_ID_Z (GPGPU_CORE_CTRL_BASE + 0x08)
#define GPGPU_CORE_CTRL_BLOCK_ID_X  (GPGPU_CORE_CTRL_BASE + 0x10)
#define GPGPU_CORE_CTRL_BLOCK_ID_Y  (GPGPU_CORE_CTRL_BASE + 0x14)
#define GPGPU_CORE_CTRL_BLOCK_ID_Z  (GPGPU_CORE_CTRL_BASE + 0x18)
#define GPGPU_CORE_CTRL_BLOCK_DIM_X (GPGPU_CORE_CTRL_BASE + 0x20)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x24)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x28)
#define GPGPU_CORE_CTRL_GRID_DIM_X  (GPGPU_CORE_CTRL_BASE + 0x30)
#define GPGPU_CORE_CTRL_GRID_DIM_Y  (GPGPU_CORE_CTRL_BASE + 0x34)
#define GPGPU_CORE_CTRL_GRID_DIM_Z  (GPGPU_CORE_CTRL_BASE + 0x38)

参数务必通过 grid_dim 和 block_dim 传递

---

### 5. 算子规范表

#### 现有算子

| 算子 | 文件 | 输入布局 | 输出布局 | Launch 参数 |
|:---|:---|:---|:---|:---|
| ReLU | `relu.S` | `[N]` @ 0x100000（原地） | 同输入 | `grid_x=N, block_x=256` |
| VectorAdd | `vector_add.S` | `[N]` @ 0x100000, `[N]` @ 0x200000 | `[N]` @ 0x300000 | `grid_x=ceil(N/256), block_x=256` |
| MatMul | `matmul.S` | `[M,K]` @ 0x100000, `[K,N]` @ 0x200000 | `[M,N]` @ 0x300000 | `grid_x=M, grid_y=K, block_x=N` |
| Conv2D | `conv2d.S` | `[H,W]` @ 0x100000, `[K,K]` @ 0x200000 | `[H-K+1,W-K+1]` @ 0x300000 | `grid_x=H, grid_y=W, block_x=K` |
| Conv2D Multi | `conv2d_multi.S` | `[C_in,H,W]` @ 0x100000, `[C_out,C_in,K,K]` @ 0x200000 | `[C_out,out_h,out_w]` @ 0x300000 | `grid_x=C_out*out_h, grid_y=W, grid_z=C_in, block_x=K, block_y=C_out` |
| MaxPool2x2 | `maxpool.S` | `[H,W]` @ 0x100000 | `[H/2,W/2]` @ 0x200000 | `grid_x=H, grid_y=W, block_x=256` |
| Softmax | `softmax_exp.S` | `[N]` @ 0x100000 | exp结果 @ 0x200000（CPU归一化）| `grid_x=ceil(N/256), block_x=256` |

#### 新增算子

| 算子 | 文件 | 输入布局 | 输出布局 | Launch 参数 |
|:---|:---|:---|:---|:---|
| BiasAdd | `bias_add.S` | `[N]` @ 0x100000（N=C\*H\*W），`[C]` @ 0x200000 | `[N]` @ 0x300000 | `grid_x=ceil(N/256), grid_y=N, grid_z=HW, block_x=256` |
| MaxPool2x2 Multi | `maxpool_multi.S` | `[C,H,W]` @ 0x100000 | `[C,H/2,W/2]` @ 0x200000 | `grid_x=out_h, grid_y=C, grid_z=H, block_x=out_w, block_y=W` |

#### BiasAdd 详细说明

```
输入:  in[N]   @ 0x100000   (N = C * H * W, 展平)
偏置:  bias[C] @ 0x200000
输出:  out[N]  @ 0x300000   out[i] = in[i] + bias[i / HW]

线程映射:
  global_id = blockIdx.x * 256 + threadIdx.x
  c         = global_id / HW

Launch:
  gpgpuLaunchKernel(dev, kernel, ceil(N/256), N, HW, 256, 1, 1, 0)
  grid_dim[0] = ceil(N/256)  (block 数，不用于计算)
  grid_dim[1] = N            (总元素数，用于边界检查)
  grid_dim[2] = HW           (每通道元素数，用于计算通道 c)
```

#### MaxPool2x2 Multi 详细说明

```
输入:  in[C, H, W]       @ 0x100000
输出:  out[C, H/2, W/2]  @ 0x200000

线程映射:
  threadIdx.x = out_x
  blockIdx.x  = out_y
  blockIdx.y  = c

Launch:
  gpgpuLaunchKernel(dev, kernel, out_h, C, H, out_w, W, 1, 0)
  grid_dim[0] = out_h  (blockIdx.x 范围)
  grid_dim[1] = C      (blockIdx.y 范围)
  grid_dim[2] = H      (传递给 kernel 的 H 值，用于计算)
  block_dim[0] = out_w (threadIdx.x 范围 = 每 block 线程数)
  block_dim[1] = W     (传递给 kernel 的 W 值，用于计算)
```

---

### 2. 数据区 (`0x100000 - 0x3FFFFF`)

| 地址 | 约定 |
|:---|:---|
| `0x100000` | **主输入**：所有算子的第一输入 |
| `0x200000` | **次输入/权重**：第二输入或权重矩阵 |
| `0x300000` | **主输出**：所有算子的输出 |

**调用约定**：算子执行前，输入必须放在 `0x100000`（和 `0x200000`）；执行后，结果在 `0x300000`。

---

### 3. 临时缓冲区 (`0x400000 - 0x7FFFFF`)

用于存放中间结果。组装 CNN 时，层与层之间需要手动搬移数据。

**示例**：
```c
// Conv1 输出在 0x300000，需要作为 ReLU 的输入
// ReLU 是原地操作，不需要搬移

// ReLU 输出还在 0x300000，需要作为 MaxPool 输入
memcpy(vram + 0x100000, vram + 0x300000, size);
run_maxpool(...);
// MaxPool 输出在 0x200000，需要搬到 0x100000 给下一层
memcpy(vram + 0x100000, vram + 0x200000, size);
```

---

### 4. 算子参数读取

/*
 * ============================================================================
 * CTRL 设备地址定义 (GPU 核心视角)
 * ============================================================================
 * GPU 核心通过访问这些地址来获取自己的线程 ID
 */
#define GPGPU_CORE_CTRL_BASE        0x80000000  /* CTRL 基地址 */
#define GPGPU_CORE_CTRL_THREAD_ID_X (GPGPU_CORE_CTRL_BASE + 0x00)
#define GPGPU_CORE_CTRL_THREAD_ID_Y (GPGPU_CORE_CTRL_BASE + 0x04)
#define GPGPU_CORE_CTRL_THREAD_ID_Z (GPGPU_CORE_CTRL_BASE + 0x08)
#define GPGPU_CORE_CTRL_BLOCK_ID_X  (GPGPU_CORE_CTRL_BASE + 0x10)
#define GPGPU_CORE_CTRL_BLOCK_ID_Y  (GPGPU_CORE_CTRL_BASE + 0x14)
#define GPGPU_CORE_CTRL_BLOCK_ID_Z  (GPGPU_CORE_CTRL_BASE + 0x18)
#define GPGPU_CORE_CTRL_BLOCK_DIM_X (GPGPU_CORE_CTRL_BASE + 0x20)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Y (GPGPU_CORE_CTRL_BASE + 0x24)
#define GPGPU_CORE_CTRL_BLOCK_DIM_Z (GPGPU_CORE_CTRL_BASE + 0x28)
#define GPGPU_CORE_CTRL_GRID_DIM_X  (GPGPU_CORE_CTRL_BASE + 0x30)
#define GPGPU_CORE_CTRL_GRID_DIM_Y  (GPGPU_CORE_CTRL_BASE + 0x34)
#define GPGPU_CORE_CTRL_GRID_DIM_Z  (GPGPU_CORE_CTRL_BASE + 0x38)

参数务必通过 grid_dim 和 block_dim 传递