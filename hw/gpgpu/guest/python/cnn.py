#!/usr/bin/env python3
"""
cnn.py — 基于 GPGPU Python API 的简单卷积神经网络

网络结构（LeNet 变体，输入 32×32 单通道图像，10 类分类）：

  Input    (1, 32, 32)
    │
  Conv1    5×5，1→4 通道，无 padding    → (4, 28, 28)
  ReLU
  MaxPool  2×2                          → (4, 14, 14)
    │
  Conv2    3×3，4→8 通道，无 padding    → (8, 12, 12)
  ReLU
  MaxPool  2×2                          → (8, 6, 6)
    │
  Flatten                               → (288,)
    │
  FC1      288 → 64                     → (64,)
  ReLU
    │
  FC2      64 → 10                      → (10,)
  Softmax                               → (10,)  ∑=1

GPGPU API 使用情况：
  gpgpu.conv2d_multi   — Conv1, Conv2
  gpgpu.relu           — 三次 ReLU（展平后调用，保持 shape 一致）
  gpgpu.maxpool2x2     — MaxPool（逐通道分别调用）
  gpgpu.matmul         — FC1, FC2
  gpgpu.softmax        — 最终归一化

运行方式：
  cd hw/gpgpu/gpgpu-driver
  make python
  export GPGPU_KERNEL_DIR=$(pwd)/bin/kernels
  sudo python3 python/cnn.py
"""

import os
import sys
import time

# 确保能找到同目录下编译的 .so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 快速检查设备
_DEV_PATH = "/dev/gpgpu0"
try:
    _fd = os.open(_DEV_PATH, os.O_RDWR)
    os.close(_fd)
except OSError as e:
    print(f"[错误] 无法打开 {_DEV_PATH}: {e}")
    print("  提示：请确认驱动已加载（make install）并以 root 运行（sudo）")
    sys.exit(1)

import numpy as np
import gpgpu


# ─────────────────────────────────────────────────────────────
# 注意事项：gpgpu.relu 的 shape 行为
# ─────────────────────────────────────────────────────────────
# relu 内部创建一个 1D 的 out_arr，但将 result->shape 设为输入 shape。
# 因此 tensor.shape 属性 ≠ tensor.numpy().shape（后者始终是 1D）。
# 解决方案：传入展平的 1D Tensor，结果 reshape 回原始 shape。
# ─────────────────────────────────────────────────────────────

def relu_gpu(x_np, dev):
    """
    对任意 shape 的 float32 numpy 数组执行 GPU ReLU，返回相同 shape。

    用法：y = relu_gpu(x_np, dev)
    """
    shape = x_np.shape
    flat  = np.ascontiguousarray(x_np.flatten(), dtype=np.float32)
    out   = gpgpu.relu(gpgpu.Tensor(flat, dev))
    return out.numpy().reshape(shape)


def maxpool_channels(x_np, dev):
    """
    对 (C, H, W) numpy 数组逐通道执行 GPU MaxPool2×2。

    返回：(C, H//2, W//2) 的 numpy 数组。
    要求：H 和 W 均为偶数。
    """
    C, H, W = x_np.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"maxpool_channels: H/W 必须为偶数，得到 ({H}, {W})")
    results = []
    for c in range(C):
        ch = np.ascontiguousarray(x_np[c], dtype=np.float32)
        out = gpgpu.maxpool2x2(gpgpu.Tensor(ch, dev))
        results.append(out.numpy())   # 每个 out 形状为 (H//2, W//2)
    return np.stack(results)          # (C, H//2, W//2)


# ─────────────────────────────────────────────────────────────
# SimpleConvNet
# ─────────────────────────────────────────────────────────────

class SimpleConvNet:
    """
    基于 GPGPU Python API 的简单 CNN（LeNet 变体）。

    输入：(1, 32, 32) 单通道图像
    输出：(num_classes,) 概率向量

    参数：
        dev         — gpgpu.Device 实例
        num_classes — 分类数（默认 10）
        seed        — 随机初始化种子（默认 42）
    """

    def __init__(self, dev, num_classes=10, seed=42):
        self.dev = dev
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)

        # ── Conv1：(C_out=4, C_in=1, K=5, K=5) ──────────────
        # 输入 (1, 32, 32) → 输出 (4, 28, 28)
        fan = 1 * 5 * 5
        self.conv1_w = rng.standard_normal((4, 1, 5, 5)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.conv1_b = np.zeros(4, dtype=np.float32)

        # ── Conv2：(C_out=8, C_in=4, K=3, K=3) ──────────────
        # 输入 (4, 14, 14) → 输出 (8, 12, 12)
        fan = 4 * 3 * 3
        self.conv2_w = rng.standard_normal((8, 4, 3, 3)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.conv2_b = np.zeros(8, dtype=np.float32)

        # ── FC1：(288, 64) ────────────────────────────────────
        # Flatten 后 8×6×6 = 288
        fan = 8 * 6 * 6   # = 288
        self.fc1_w = rng.standard_normal((fan, 64)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.fc1_b = np.zeros(64, dtype=np.float32)

        # ── FC2：(64, num_classes) ────────────────────────────
        fan = 64
        self.fc2_w = rng.standard_normal((64, num_classes)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.fc2_b = np.zeros(num_classes, dtype=np.float32)

    # ─── 前向推断 ──────────────────────────────────────────────

    def forward(self, x_np):
        """
        单张图像前向推断。

        参数：
            x_np — numpy.ndarray，shape (1, 32, 32) 或 (32, 32)，float32

        返回：
            probs — numpy.ndarray，shape (num_classes,)，概率分布，∑≈1
        """
        dev = self.dev

        # 输入格式规范化
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :]         # (H,W) → (1,H,W)
        x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        if x_np.shape != (1, 32, 32):
            raise ValueError(f"期望输入 shape (1,32,32)，得到 {x_np.shape}")

        # ══════════════════════════════════════════════
        # Conv 层 1：(1,32,32) → conv → (4,28,28)
        # ══════════════════════════════════════════════
        inp_t = gpgpu.Tensor(x_np, dev)
        w1_t  = gpgpu.Tensor(self.conv1_w, dev)
        # GPU 多通道卷积：output shape (4, 28, 28)
        conv1_np = gpgpu.conv2d_multi(inp_t, w1_t, c_in=1, c_out=4).numpy()
        conv1_np = conv1_np.reshape(4, 28, 28)
        # 加偏置（广播，CPU）
        conv1_np += self.conv1_b[:, np.newaxis, np.newaxis]
        # GPU ReLU
        relu1_np = relu_gpu(conv1_np, dev)                 # (4, 28, 28)
        # GPU MaxPool2×2：(4,28,28) → (4,14,14)
        pool1_np = maxpool_channels(relu1_np, dev)          # (4, 14, 14)

        # ══════════════════════════════════════════════
        # Conv 层 2：(4,14,14) → conv → (8,12,12)
        # ══════════════════════════════════════════════
        p1_t = gpgpu.Tensor(pool1_np, dev)
        w2_t = gpgpu.Tensor(self.conv2_w, dev)
        # GPU 多通道卷积：output shape (8, 12, 12)
        conv2_np = gpgpu.conv2d_multi(p1_t, w2_t, c_in=4, c_out=8).numpy()
        conv2_np = conv2_np.reshape(8, 12, 12)
        # 加偏置（CPU）
        conv2_np += self.conv2_b[:, np.newaxis, np.newaxis]
        # GPU ReLU
        relu2_np = relu_gpu(conv2_np, dev)                 # (8, 12, 12)
        # GPU MaxPool2×2：(8,12,12) → (8,6,6)
        pool2_np = maxpool_channels(relu2_np, dev)          # (8, 6, 6)

        # ══════════════════════════════════════════════
        # Flatten + FC 层 1：(1,288) → matmul → (1,64)
        # ══════════════════════════════════════════════
        # (8,6,6) → (1,288)
        flat_np = np.ascontiguousarray(pool2_np.flatten()[np.newaxis, :], dtype=np.float32)
        flat_t  = gpgpu.Tensor(flat_np, dev)
        w3_t    = gpgpu.Tensor(self.fc1_w, dev)
        # GPU 矩阵乘：(1,288) @ (288,64) = (1,64)
        fc1_np  = gpgpu.matmul(flat_t, w3_t).numpy()      # (1, 64)
        # 加偏置（CPU）
        fc1_np  += self.fc1_b[np.newaxis, :]
        # GPU ReLU
        relu3_np = relu_gpu(fc1_np, dev)                   # (1, 64)

        # ══════════════════════════════════════════════
        # FC 层 2：(1,64) → matmul → (1,num_classes)
        # ══════════════════════════════════════════════
        relu3_t  = gpgpu.Tensor(relu3_np, dev)
        w4_t     = gpgpu.Tensor(self.fc2_w, dev)
        # GPU 矩阵乘：(1,64) @ (64,num_classes) = (1,num_classes)
        fc2_np   = gpgpu.matmul(relu3_t, w4_t).numpy()    # (1, num_classes)
        # 加偏置（CPU）并展平
        logits   = fc2_np.flatten() + self.fc2_b          # (num_classes,)

        # ══════════════════════════════════════════════
        # GPU Softmax → 概率分布
        # ══════════════════════════════════════════════
        logits_t = gpgpu.Tensor(np.ascontiguousarray(logits, dtype=np.float32), dev)
        probs    = gpgpu.softmax(logits_t)

        return probs.numpy()   # (num_classes,)

    # ─── 统计与显示 ────────────────────────────────────────────

    def num_parameters(self):
        """返回模型总可训练参数量"""
        return sum(p.size for p in [
            self.conv1_w, self.conv1_b,
            self.conv2_w, self.conv2_b,
            self.fc1_w,   self.fc1_b,
            self.fc2_w,   self.fc2_b,
        ])

    def summary(self):
        """打印网络结构摘要"""
        COL = 56
        print("=" * COL)
        print("  SimpleConvNet 结构摘要")
        print("=" * COL)
        rows = [
            ("Input",      "(1, 32, 32)",          "—",     "—"),
            ("Conv1",      "(4, 28, 28)",           "5×5",   self.conv1_w.size + self.conv1_b.size),
            ("ReLU",       "(4, 28, 28)",           "—",     "—"),
            ("MaxPool2×2", "(4, 14, 14)",           "—",     "—"),
            ("Conv2",      "(8, 12, 12)",           "3×3",   self.conv2_w.size + self.conv2_b.size),
            ("ReLU",       "(8, 12, 12)",           "—",     "—"),
            ("MaxPool2×2", "(8,  6,  6)",           "—",     "—"),
            ("Flatten",    "(288,)",                "—",     "—"),
            ("FC1",        "(64,)",                 "Dense", self.fc1_w.size + self.fc1_b.size),
            ("ReLU",       "(64,)",                 "—",     "—"),
            ("FC2",        f"({self.num_classes},)","Dense", self.fc2_w.size + self.fc2_b.size),
            ("Softmax",    f"({self.num_classes},)","—",     "—"),
        ]
        print(f"  {'层名':<12} {'输出形状':<14} {'核/类型':<8} {'参数量':>7}")
        print("  " + "-" * (COL - 2))
        for name, shape, ktype, params in rows:
            p = str(params) if isinstance(params, int) else params
            print(f"  {name:<12} {shape:<14} {ktype:<8} {p:>7}")
        print("  " + "-" * (COL - 2))
        print(f"  {'总参数量':<36} {self.num_parameters():>7}")
        print("=" * COL)


# ─────────────────────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────────────────────

def verify_forward(net):
    """
    验证前向推断的正确性：
      1. 全零输入 → 输出应为均匀分布（偏置均为零）
      2. 任意输入 → 概率之和应≈1
      3. 所有输出值 ∈ [0, 1]
    """
    print("\n验证推断正确性...")
    PASS = "  [PASS]"
    FAIL = "  [FAIL]"

    # 1. 全零输入
    x_zero = np.zeros((1, 32, 32), dtype=np.float32)
    p = net.forward(x_zero)
    prob_sum = float(p.sum())
    ok = abs(prob_sum - 1.0) < 0.01
    print(f"{PASS if ok else FAIL} 全零输入：概率和 = {prob_sum:.6f}")
    assert ok, "概率和偏差过大"

    # 2. 随机输入×3，检查概率和与值域
    rng = np.random.default_rng(99)
    for i in range(3):
        x = rng.standard_normal((1, 32, 32)).astype(np.float32) * 0.3
        p = net.forward(x)
        sum_ok  = abs(float(p.sum()) - 1.0) < 0.01
        range_ok = bool((p >= 0).all() and (p <= 1).all())
        ok = sum_ok and range_ok
        print(f"{PASS if ok else FAIL} 随机输入 {i+1}：概率和 = {float(p.sum()):.6f}，"
              f"max = {float(p.max()):.4f}，pred = {int(p.argmax())}")
        assert ok

    print("  所有验证通过！")


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    # ── 设备初始化 ──
    print("正在初始化 GPGPU 设备...")
    dev = gpgpu.Device()
    print("设备就绪\n")

    # ── 创建网络并打印结构 ──
    net = SimpleConvNet(dev, num_classes=10)
    net.summary()

    # ── 正确性验证 ──
    verify_forward(net)

    # ── 推断演示 ──
    CLASS_NAMES = [str(i) for i in range(10)]
    print("\n" + "=" * 56)
    print("  推断演示（3 张随机合成图像）")
    print("=" * 56)

    rng = np.random.default_rng(0)
    total_time = 0.0

    for idx in range(3):
        # 模拟归一化后的灰度图像（均值≈0，标准差≈0.3）
        img = rng.standard_normal((1, 32, 32)).astype(np.float32) * 0.3

        t0 = time.time()
        probs = net.forward(img)
        elapsed = time.time() - t0
        total_time += elapsed

        pred = int(probs.argmax())
        bar  = "".join(
            "█" * max(1, round(p * 20)) if i == pred else "░" * max(1, round(p * 20))
            for i, p in enumerate(probs)
        )
        print(f"\n  图像 {idx + 1}：")
        print(f"    预测类别  : {CLASS_NAMES[pred]}")
        print(f"    预测置信度: {probs[pred]:.4f}")
        print(f"    概率分布  : [{' '.join(f'{p:.3f}' for p in probs)}]")
        print(f"    推断耗时  : {elapsed * 1000:.1f} ms")

    print(f"\n  平均推断耗时：{total_time / 3 * 1000:.1f} ms")
    print("\n完成！")


if __name__ == "__main__":
    main()
