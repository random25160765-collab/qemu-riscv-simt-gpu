#!/usr/bin/env python3
"""
GPGPU Python API 集成测试

运行方式：
    cd hw/gpgpu/gpgpu-driver
    make python
    export GPGPU_KERNEL_DIR=$(pwd)/bin/kernels
    sudo python3 python/test_gpgpu.py
"""

import os
import sys

# 确保能找到同目录下编译的 .so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 先检查 /dev/gpgpu0 是否可访问（快速诊断）
_DEV = "/dev/gpgpu0"
try:
    _fd = os.open(_DEV, os.O_RDWR)
    os.close(_fd)
except OSError as e:
    print(f"[错误] 无法打开 {_DEV}: {e}")
    print(f"  提示：请确认驱动已加载（make install）并以 root 运行（sudo）")
    sys.exit(1)

import numpy as np
import gpgpu


# ============================================================
# 测试工具
# ============================================================

def passed(name):
    print(f"  [PASS] {name}")

def failed(name, reason):
    print(f"  [FAIL] {name}: {reason}")
    sys.exit(1)


# ============================================================
# 各项测试
# ============================================================

def test_device():
    print("=== 设备初始化 ===")
    dev = gpgpu.Device()
    assert repr(dev) == "<gpgpu.Device>"
    passed("Device()")
    return dev


def test_tensor_create(dev):
    print("\n=== Tensor 创建 ===")

    # 基本创建
    a = gpgpu.Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), dev)
    assert a.shape == (4,), f"shape 期望 (4,)，得到 {a.shape}"
    assert a.ndim == 1
    assert a.numel() == 4
    passed("1D Tensor 创建")

    # 自动 float64 → float32 转换
    b = gpgpu.Tensor(np.array([1.0, 2.0]), dev)   # float64 输入
    assert b.shape == (2,)
    passed("float64 自动转 float32")

    # 2D tensor
    c = gpgpu.Tensor(np.ones((3, 4), dtype=np.float32), dev)
    assert c.shape == (3, 4)
    assert c.ndim == 2
    assert c.numel() == 12
    passed("2D Tensor 创建")

    # repr
    assert "float32" in repr(a)
    passed("__repr__")


def test_reshape(dev):
    print("\n=== reshape ===")
    x = gpgpu.Tensor(np.arange(12, dtype=np.float32), dev)
    y = x.reshape([3, 4])
    assert y.shape == (3, 4), f"期望 (3,4)，得到 {y.shape}"
    assert y.numel() == 12
    passed("1D → 2D reshape")

    z = y.reshape([2, 6])
    assert z.shape == (2, 6)
    passed("2D → 2D reshape")

    # 元素数不匹配应报错
    try:
        x.reshape([2, 7])
        failed("reshape 错误检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("reshape 元素数不匹配错误")


def test_numpy_roundtrip(dev):
    print("\n=== numpy 转换 ===")
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = gpgpu.Tensor(data, dev)
    out = t.numpy()
    assert np.array_equal(out, data), f"roundtrip 失败：{out}"
    passed("Tensor → numpy 零拷贝")


def test_vecadd(dev):
    print("\n=== VecAdd ===")
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    a = gpgpu.Tensor(a_np, dev)
    b = gpgpu.Tensor(b_np, dev)

    # 函数式接口
    c = gpgpu.add(a, b)
    assert np.allclose(c.numpy(), a_np + b_np, atol=1e-4), \
        f"add 结果不正确：{c.numpy()}"
    passed("gpgpu.add()")

    # 运算符重载
    c2 = a + b
    assert np.allclose(c2.numpy(), a_np + b_np, atol=1e-4)
    passed("a + b 运算符")

    # shape 不匹配应报错
    c_short = gpgpu.Tensor(np.array([1.0, 2.0], dtype=np.float32), dev)
    try:
        gpgpu.add(a, c_short)
        failed("add shape 错误检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("add shape 不匹配错误")


def test_relu(dev):
    print("\n=== ReLU ===")
    x_np = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    x = gpgpu.Tensor(x_np, dev)
    y = gpgpu.relu(x)
    expected = np.maximum(0.0, x_np)
    assert np.allclose(y.numpy(), expected, atol=1e-4), \
        f"relu 结果不正确：{y.numpy()}"
    passed("relu 计算正确")

    # 确保输入未被修改（不可变语义）
    assert np.allclose(x.numpy(), x_np), "relu 不应修改输入"
    passed("relu 不修改输入")


def test_matmul(dev):
    print("\n=== MatMul ===")
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    a = gpgpu.Tensor(a_np, dev)
    b = gpgpu.Tensor(b_np, dev)

    c = gpgpu.matmul(a, b)
    expected = a_np @ b_np
    assert c.shape == (2, 2), f"matmul shape 期望 (2,2)，得到 {c.shape}"
    assert np.allclose(c.numpy(), expected, atol=0.1), \
        f"matmul 结果不正确：\n{c.numpy()}\n期望：\n{expected}"
    passed("gpgpu.matmul()")

    # @ 运算符
    c2 = a @ b
    assert np.allclose(c2.numpy(), expected, atol=0.1)
    passed("a @ b 运算符")

    # 内维度不匹配应报错
    d = gpgpu.Tensor(np.ones((3, 2), dtype=np.float32), dev)
    try:
        gpgpu.matmul(a, d)
        failed("matmul 维度错误检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("matmul 内维度不匹配错误")


def test_softmax(dev):
    print("\n=== Softmax ===")
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = gpgpu.Tensor(x_np, dev)
    y = gpgpu.softmax(x)
    result = y.numpy()

    assert abs(result.sum() - 1.0) < 0.01, \
        f"softmax 和期望为 1，实际为 {result.sum()}"
    assert (result >= 0).all(), "softmax 输出应非负"
    assert result.argmax() == x_np.argmax(), \
        "softmax 最大值位置应与输入一致"
    passed("softmax 结果正确")


def test_maxpool2x2(dev):
    print("\n=== MaxPool 2×2 ===")
    x_np = np.array([
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 7.0, 6.0, 8.0],
        [9.0, 11.0, 10.0, 12.0],
        [13.0, 15.0, 14.0, 16.0],
    ], dtype=np.float32)
    x = gpgpu.Tensor(x_np, dev)
    y = gpgpu.maxpool2x2(x)

    assert y.shape == (2, 2), f"maxpool2x2 shape 期望 (2,2)，得到 {y.shape}"
    result = y.numpy()
    # 每个 2×2 窗口的最大值：7, 8, 15, 16
    expected = np.array([[7.0, 8.0], [15.0, 16.0]], dtype=np.float32)
    assert np.allclose(result, expected, atol=0.1), \
        f"maxpool2x2 结果不正确：{result}"
    passed("maxpool2x2 结果正确")

    # H/W 为奇数应报错
    x_odd = gpgpu.Tensor(np.ones((3, 4), dtype=np.float32), dev)
    try:
        gpgpu.maxpool2x2(x_odd)
        failed("maxpool2x2 奇数尺寸检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("maxpool2x2 奇数 H 报错")


def test_conv2d(dev):
    print("\n=== Conv2D（单通道）===")
    # 全 1 输入 + 全 1 kernel → 每个输出 = K*K = 9
    inp_np = np.ones((5, 5), dtype=np.float32)
    w_np   = np.ones((3, 3), dtype=np.float32)
    inp    = gpgpu.Tensor(inp_np, dev)
    weight = gpgpu.Tensor(w_np, dev)
    out    = gpgpu.conv2d(inp, weight)

    assert out.shape == (3, 3), f"conv2d shape 期望 (3,3)，得到 {out.shape}"
    assert np.allclose(out.numpy(), 9.0, atol=0.1), \
        f"conv2d 全1输入应输出 9，得到：{out.numpy()}"
    passed("conv2d 结果正确（全1测试）")

    # 非方形 kernel 应报错
    w_rect = gpgpu.Tensor(np.ones((2, 3), dtype=np.float32), dev)
    try:
        gpgpu.conv2d(inp, w_rect)
        failed("conv2d 非方形 kernel 检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("conv2d 非方形 kernel 报错")


def test_error_same_device():
    print("\n=== 跨设备错误检测 ===")
    dev1 = gpgpu.Device()
    dev2 = gpgpu.Device()

    a = gpgpu.Tensor(np.array([1.0, 2.0], dtype=np.float32), dev1)
    b = gpgpu.Tensor(np.array([3.0, 4.0], dtype=np.float32), dev2)
    try:
        gpgpu.add(a, b)
        failed("跨设备 add 检测", "应抛出异常")
    except (ValueError, RuntimeError):
        passed("不同设备 add 报错")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("GPGPU Python API 集成测试")
    print("=" * 50)

    try:
        dev = test_device()
        test_tensor_create(dev)
        test_reshape(dev)
        test_numpy_roundtrip(dev)
        test_vecadd(dev)
        test_relu(dev)
        test_matmul(dev)
        test_softmax(dev)
        test_maxpool2x2(dev)
        test_conv2d(dev)
        test_error_same_device()

        print("\n" + "=" * 50)
        print("所有测试通过！")
        print("=" * 50)

    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
