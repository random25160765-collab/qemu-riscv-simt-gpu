#!/usr/bin/env python3
"""
cnn.py — 基于 GPGPU Python API 的简单卷积神经网络

网络结构（LeNet 变体，输入 32×32 单通道图像，10 类分类）：

  Input    (1, 32, 32)
    │
  Conv1    5×5，1→4 通道，无 padding    → (4, 28, 28)
  BiasAdd                               → (4, 28, 28)
  ReLU
  MaxPool  2×2                          → (4, 14, 14)
    │
  Conv2    3×3，4→8 通道，无 padding    → (8, 12, 12)
  BiasAdd                               → (8, 12, 12)
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
  gpgpu.conv2d_multi      — Conv1, Conv2（C API 内部逐通道累加）
  gpgpu.bias_add          — 偏置加法（GPU 广播，单次调用）
  gpgpu.relu              — 三次 ReLU
  gpgpu.maxpool2x2_multi  — MaxPool（单次 GPU 调用处理所有通道）
  gpgpu.matmul            — FC1, FC2
  gpgpu.softmax           — 最终归一化

运行方式：
  cd hw/gpgpu/guest
  make python
  export GPGPU_KERNEL_DIR=$(pwd)/bin/kernels
  sudo python3 python/cnn.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def relu_gpu(x_np, dev):
    """对任意 shape 的 float32 numpy 数组执行 GPU ReLU，返回相同 shape。"""
    shape = x_np.shape
    flat  = np.ascontiguousarray(x_np.flatten(), dtype=np.float32)
    out   = gpgpu.relu(gpgpu.Tensor(flat, dev))
    return out.numpy().reshape(shape)


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

        fan = 1 * 5 * 5
        self.conv1_w = rng.standard_normal((4, 1, 5, 5)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.conv1_b = np.zeros(4, dtype=np.float32)

        fan = 4 * 3 * 3
        self.conv2_w = rng.standard_normal((8, 4, 3, 3)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.conv2_b = np.zeros(8, dtype=np.float32)

        fan = 8 * 6 * 6
        self.fc1_w = rng.standard_normal((fan, 64)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.fc1_b = np.zeros(64, dtype=np.float32)

        fan = 64
        self.fc2_w = rng.standard_normal((64, num_classes)).astype(np.float32) * np.sqrt(2.0 / fan)
        self.fc2_b = np.zeros(num_classes, dtype=np.float32)

    def forward(self, x_np):
        """
        单张图像前向推断。

        参数：
            x_np — numpy.ndarray，shape (1, 32, 32) 或 (32, 32)，float32

        返回：
            probs — numpy.ndarray，shape (num_classes,)，概率分布，∑≈1
        """
        dev = self.dev

        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :]
        x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        if x_np.shape != (1, 32, 32):
            raise ValueError(f"期望输入 shape (1,32,32)，得到 {x_np.shape}")

        # ══ Conv1：(1,32,32) → (4,28,28) ══
        inp_t = gpgpu.Tensor(x_np, dev)
        w1_t  = gpgpu.Tensor(self.conv1_w, dev)
        conv1 = gpgpu.conv2d_multi(inp_t, w1_t, c_in=1, c_out=4)  # (4,28,28)
        # GPU BiasAdd（广播，单次调用）
        b1_t  = gpgpu.Tensor(self.conv1_b, dev)
        conv1 = gpgpu.bias_add(conv1, b1_t)                        # (4,28,28)
        # GPU ReLU
        conv1_np = relu_gpu(conv1.numpy().reshape(4, 28, 28), dev) # (4,28,28)
        # GPU MaxPool（单次调用所有通道）
        pool1_t  = gpgpu.Tensor(np.ascontiguousarray(conv1_np), dev)
        pool1    = gpgpu.maxpool2x2_multi(pool1_t)                  # (4,14,14)

        # ══ Conv2：(4,14,14) → (8,12,12) ══
        w2_t  = gpgpu.Tensor(self.conv2_w, dev)
        conv2 = gpgpu.conv2d_multi(pool1, w2_t, c_in=4, c_out=8)  # (8,12,12)
        b2_t  = gpgpu.Tensor(self.conv2_b, dev)
        conv2 = gpgpu.bias_add(conv2, b2_t)                        # (8,12,12)
        conv2_np = relu_gpu(conv2.numpy().reshape(8, 12, 12), dev) # (8,12,12)
        pool2_t  = gpgpu.Tensor(np.ascontiguousarray(conv2_np), dev)
        pool2    = gpgpu.maxpool2x2_multi(pool2_t)                  # (8,6,6)

        # ══ Flatten + FC1：(1,288) @ (288,64) = (1,64) ══
        flat_np = np.ascontiguousarray(pool2.numpy().flatten()[np.newaxis, :], dtype=np.float32)
        flat_t  = gpgpu.Tensor(flat_np, dev)
        w3_t    = gpgpu.Tensor(self.fc1_w, dev)
        fc1_np  = gpgpu.matmul(flat_t, w3_t).numpy()              # (1,64)
        fc1_np  += self.fc1_b[np.newaxis, :]                      # CPU bias（1D，快速）
        relu3_np = relu_gpu(fc1_np, dev)                           # (1,64)

        # ══ FC2：(1,64) @ (64,num_classes) = (1,num_classes) ══
        relu3_t = gpgpu.Tensor(relu3_np, dev)
        w4_t    = gpgpu.Tensor(self.fc2_w, dev)
        fc2_np  = gpgpu.matmul(relu3_t, w4_t).numpy()             # (1,num_classes)
        logits  = fc2_np.flatten() + self.fc2_b                   # (num_classes,)

        # ══ GPU Softmax ══
        logits_t = gpgpu.Tensor(np.ascontiguousarray(logits, dtype=np.float32), dev)
        return gpgpu.softmax(logits_t).numpy()                     # (num_classes,)

    def num_parameters(self):
        return sum(p.size for p in [
            self.conv1_w, self.conv1_b,
            self.conv2_w, self.conv2_b,
            self.fc1_w,   self.fc1_b,
            self.fc2_w,   self.fc2_b,
        ])

    def summary(self):
        COL = 56
        print("=" * COL)
        print("  SimpleConvNet 结构摘要")
        print("=" * COL)
        rows = [
            ("Input",      "(1, 32, 32)",          "—",     "—"),
            ("Conv1",      "(4, 28, 28)",           "5×5",   self.conv1_w.size + self.conv1_b.size),
            ("BiasAdd",    "(4, 28, 28)",           "GPU",   "—"),
            ("ReLU",       "(4, 28, 28)",           "—",     "—"),
            ("MaxPool2×2", "(4, 14, 14)",           "multi", "—"),
            ("Conv2",      "(8, 12, 12)",           "3×3",   self.conv2_w.size + self.conv2_b.size),
            ("BiasAdd",    "(8, 12, 12)",           "GPU",   "—"),
            ("ReLU",       "(8, 12, 12)",           "—",     "—"),
            ("MaxPool2×2", "(8,  6,  6)",           "multi", "—"),
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


def verify_forward(net):
    print("\n验证推断正确性...")
    PASS = "  [PASS]"
    FAIL = "  [FAIL]"

    x_zero = np.zeros((1, 32, 32), dtype=np.float32)
    p = net.forward(x_zero)
    prob_sum = float(p.sum())
    ok = abs(prob_sum - 1.0) < 0.01
    print(f"{PASS if ok else FAIL} 全零输入：概率和 = {prob_sum:.6f}")
    assert ok, "概率和偏差过大"

    rng = np.random.default_rng(99)
    for i in range(3):
        x = rng.standard_normal((1, 32, 32)).astype(np.float32) * 0.3
        p = net.forward(x)
        sum_ok   = abs(float(p.sum()) - 1.0) < 0.01
        range_ok = bool((p >= 0).all() and (p <= 1).all())
        ok = sum_ok and range_ok
        print(f"{PASS if ok else FAIL} 随机输入 {i+1}：概率和 = {float(p.sum()):.6f}，"
              f"max = {float(p.max()):.4f}，pred = {int(p.argmax())}")
        assert ok

    print("  所有验证通过！")


def main():
    print("正在初始化 GPGPU 设备...")
    dev = gpgpu.Device()
    print("设备就绪\n")

    net = SimpleConvNet(dev, num_classes=10)
    net.summary()

    verify_forward(net)

    CLASS_NAMES = [str(i) for i in range(10)]
    print("\n" + "=" * 56)
    print("  推断演示（3 张随机合成图像）")
    print("=" * 56)

    rng = np.random.default_rng(0)
    total_time = 0.0

    for idx in range(3):
        img = rng.standard_normal((1, 32, 32)).astype(np.float32) * 0.3
        t0 = time.time()
        probs = net.forward(img)
        elapsed = time.time() - t0
        total_time += elapsed

        pred = int(probs.argmax())
        print(f"\n  图像 {idx + 1}：")
        print(f"    预测类别  : {CLASS_NAMES[pred]}")
        print(f"    预测置信度: {probs[pred]:.4f}")
        print(f"    概率分布  : [{' '.join(f'{p:.3f}' for p in probs)}]")
        print(f"    推断耗时  : {elapsed * 1000:.1f} ms")

    print(f"\n  平均推断耗时：{total_time / 3 * 1000:.1f} ms")
    print("\n完成！")


if __name__ == "__main__":
    main()

