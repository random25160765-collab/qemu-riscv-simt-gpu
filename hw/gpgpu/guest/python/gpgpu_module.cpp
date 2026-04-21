// python/gpgpu_module.cpp
//
// pybind11 绑定：为 GPGPU 运行时提供 PyTorch-like Python API
//
// 用法示例：
//   import gpgpu, numpy as np
//   dev = gpgpu.Device()
//   a = gpgpu.Tensor(np.array([1,2,3,4], dtype=np.float32), dev)
//   b = gpgpu.Tensor(np.array([5,6,7,8], dtype=np.float32), dev)
//   c = a + b
//   print(c.numpy())   # [6. 8. 10. 12.]

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "../runtime/gpgpu_runtime.h"
}

namespace py = pybind11;

// ============================================================
// 错误处理
// ============================================================

static void check_error(GPGPUError err, const char *ctx = "") {
    if (err == GPGPU_SUCCESS) return;

    std::string msg;
    if (ctx && ctx[0]) { msg = ctx; msg += ": "; }
    msg += gpgpuGetErrorString(err);

    // 附加 errno 信息（帮助诊断 open/mmap 失败原因）
    if (errno != 0) {
        msg += " (errno=";
        msg += std::to_string(errno);
        msg += ": ";
        msg += strerror(errno);
        msg += ")";
    }

    switch (err) {
        case GPGPU_ERROR_INVALID_VALUE:
            throw std::invalid_argument(msg);
        case GPGPU_ERROR_OUT_OF_MEMORY:
            throw std::bad_alloc();
        default:
            throw std::runtime_error(msg);
    }
}

// ============================================================
// DeviceWrapper
// ============================================================

struct DeviceWrapper {
    GPGPUDevice    dev  = nullptr;
    GPGPUOperators ops  = {};

    DeviceWrapper() {
        check_error(gpgpuInit(&dev),             "gpgpuInit");
        check_error(gpgpuReset(dev),             "gpgpuReset");
        check_error(gpgpuLoadOperators(dev, &ops), "gpgpuLoadOperators");
    }

    ~DeviceWrapper() {
        if (dev) { gpgpuClose(dev); dev = nullptr; }
    }

    DeviceWrapper(const DeviceWrapper &) = delete;
    DeviceWrapper &operator=(const DeviceWrapper &) = delete;
};

// ============================================================
// TensorWrapper
// ============================================================

struct TensorWrapper {
    // CPU 侧存储（连续 float32 numpy 数组）
    py::array_t<float, py::array::c_style | py::array::forcecast> data;
    std::vector<ssize_t> shape;
    std::shared_ptr<DeviceWrapper> device;

    TensorWrapper(py::array_t<float> arr, std::shared_ptr<DeviceWrapper> dev)
        : device(std::move(dev))
    {
        data  = py::array_t<float, py::array::c_style | py::array::forcecast>(arr);
        auto  info = data.request();
        shape = std::vector<ssize_t>(info.shape.begin(), info.shape.end());
    }

    const float *ptr() const {
        return static_cast<const float *>(data.request().ptr);
    }

    float *mutable_ptr() {
        return static_cast<float *>(data.request().ptr);
    }

    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    py::array_t<float> numpy() const { return data; }

    void assert_1d(const char *op) const {
        if (shape.size() != 1)
            throw std::invalid_argument(
                std::string(op) + " 需要 1D tensor，但收到 " +
                std::to_string(shape.size()) + "D");
    }

    void assert_2d(const char *op) const {
        if (shape.size() != 2)
            throw std::invalid_argument(
                std::string(op) + " 需要 2D tensor，但收到 " +
                std::to_string(shape.size()) + "D");
    }

    std::string shape_str() const {
        std::string s = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        return s + ")";
    }
};

// ============================================================
// 辅助：检查同一设备
// ============================================================

static void check_same_device(const TensorWrapper &a,
                               const TensorWrapper &b,
                               const char *op)
{
    if (a.device.get() != b.device.get())
        throw std::invalid_argument(
            std::string(op) + ": 两个 tensor 必须在同一个设备上");
}

// ============================================================
// 算子实现
// ============================================================

// --- add：C = A + B（逐元素，要求 shape 一致）---
static std::shared_ptr<TensorWrapper>
op_add(std::shared_ptr<TensorWrapper> a,
       std::shared_ptr<TensorWrapper> b)
{
    check_same_device(*a, *b, "add");
    if (a->shape != b->shape)
        throw std::invalid_argument(
            "add: shape 不匹配 " + a->shape_str() + " vs " + b->shape_str());

    size_t n = a->numel();
    py::array_t<float> out_arr(static_cast<ssize_t>(n));
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuVecAdd(a->device->dev, a->device->ops.vecadd,
                             a->ptr(), b->ptr(), out_ptr,
                             static_cast<uint32_t>(n)),
                "gpgpuVecAdd");

    auto result = std::make_shared<TensorWrapper>(out_arr, a->device);
    result->shape = a->shape;
    return result;
}

// --- relu：y = max(0, x)，返回新 Tensor（不修改输入）---
static std::shared_ptr<TensorWrapper>
op_relu(std::shared_ptr<TensorWrapper> x)
{
    size_t n = x->numel();
    py::array_t<float> out_arr(static_cast<ssize_t>(n));
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);
    std::memcpy(out_ptr, x->ptr(), n * sizeof(float));  // 拷贝一份，原地执行

    check_error(gpgpuReLU(x->device->dev, x->device->ops.relu,
                           out_ptr, static_cast<uint32_t>(n)),
                "gpgpuReLU");

    auto result = std::make_shared<TensorWrapper>(out_arr, x->device);
    result->shape = x->shape;
    return result;
}

// --- matmul：C(M,N) = A(M,K) * B(K,N)，要求 2D ---
static std::shared_ptr<TensorWrapper>
op_matmul(std::shared_ptr<TensorWrapper> a,
          std::shared_ptr<TensorWrapper> b)
{
    check_same_device(*a, *b, "matmul");
    a->assert_2d("matmul(a)");
    b->assert_2d("matmul(b)");

    uint32_t M  = static_cast<uint32_t>(a->shape[0]);
    uint32_t K  = static_cast<uint32_t>(a->shape[1]);
    uint32_t K2 = static_cast<uint32_t>(b->shape[0]);
    uint32_t N  = static_cast<uint32_t>(b->shape[1]);

    if (K != K2)
        throw std::invalid_argument(
            "matmul: 内维度不匹配 " + a->shape_str() + " vs " + b->shape_str());

    std::vector<ssize_t> out_shape = {static_cast<ssize_t>(M),
                                       static_cast<ssize_t>(N)};
    py::array_t<float> out_arr(out_shape);
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuMatMul(a->device->dev, a->device->ops.matmul,
                             a->ptr(), b->ptr(), out_ptr,
                             M, N, K),
                "gpgpuMatMul");

    auto result = std::make_shared<TensorWrapper>(out_arr, a->device);
    result->shape = out_shape;
    return result;
}

// --- softmax：y_i = exp(x_i) / sum(exp(x_j))，要求 1D ---
static std::shared_ptr<TensorWrapper>
op_softmax(std::shared_ptr<TensorWrapper> x)
{
    x->assert_1d("softmax");
    uint32_t n = static_cast<uint32_t>(x->shape[0]);

    py::array_t<float> out_arr(static_cast<ssize_t>(n));
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuSoftmax(x->device->dev, x->device->ops.softmax,
                              x->ptr(), out_ptr, n),
                "gpgpuSoftmax");

    return std::make_shared<TensorWrapper>(out_arr, x->device);
}

// --- maxpool2x2：MaxPool 2×2，要求 2D，H/W 为偶数 ---
static std::shared_ptr<TensorWrapper>
op_maxpool2x2(std::shared_ptr<TensorWrapper> x)
{
    x->assert_2d("maxpool2x2");
    uint32_t H = static_cast<uint32_t>(x->shape[0]);
    uint32_t W = static_cast<uint32_t>(x->shape[1]);

    if (H % 2 != 0 || W % 2 != 0)
        throw std::invalid_argument(
            "maxpool2x2: H 和 W 必须为偶数，但收到 " + x->shape_str());

    uint32_t out_h = H / 2, out_w = W / 2;
    std::vector<ssize_t> out_shape = {static_cast<ssize_t>(out_h),
                                       static_cast<ssize_t>(out_w)};
    py::array_t<float> out_arr(out_shape);
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuMaxPool2x2(x->device->dev, x->device->ops.maxpool,
                                 x->ptr(), out_ptr, H, W),
                "gpgpuMaxPool2x2");

    auto result = std::make_shared<TensorWrapper>(out_arr, x->device);
    result->shape = out_shape;
    return result;
}

// --- conv2d：单通道卷积，要求 2D，kernel 为方形 ---
static std::shared_ptr<TensorWrapper>
op_conv2d(std::shared_ptr<TensorWrapper> inp,
          std::shared_ptr<TensorWrapper> weight)
{
    check_same_device(*inp, *weight, "conv2d");
    inp->assert_2d("conv2d(input)");
    weight->assert_2d("conv2d(weight)");

    uint32_t H  = static_cast<uint32_t>(inp->shape[0]);
    uint32_t W  = static_cast<uint32_t>(inp->shape[1]);
    uint32_t Kh = static_cast<uint32_t>(weight->shape[0]);
    uint32_t Kw = static_cast<uint32_t>(weight->shape[1]);

    if (Kh != Kw)
        throw std::invalid_argument(
            "conv2d: kernel 必须为方形，但收到 " + weight->shape_str());
    if (H < Kh || W < Kw)
        throw std::invalid_argument(
            "conv2d: 输入 " + inp->shape_str() + " 小于 kernel " + weight->shape_str());

    uint32_t out_h = H - Kh + 1, out_w = W - Kw + 1;
    std::vector<ssize_t> out_shape = {static_cast<ssize_t>(out_h),
                                       static_cast<ssize_t>(out_w)};
    py::array_t<float> out_arr(out_shape);
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuConv2D(inp->device->dev, inp->device->ops.conv2d,
                             inp->ptr(), weight->ptr(), out_ptr,
                             H, W, Kh),
                "gpgpuConv2D");

    auto result = std::make_shared<TensorWrapper>(out_arr, inp->device);
    result->shape = out_shape;
    return result;
}

// --- conv2d_multi：多通道卷积
//   inp:    3D (C_IN, H, W)
//   weight: 4D (C_OUT, C_IN, K, K)
//   output: 3D (C_OUT, H-K+1, W-K+1)
//
//   注意：C API gpgpuConv2DMulti 内部逐通道调用单通道 kernel，
//         因此传 ops.conv2d 而非 ops.conv2d_multi
static std::shared_ptr<TensorWrapper>
op_conv2d_multi(std::shared_ptr<TensorWrapper> inp,
                std::shared_ptr<TensorWrapper> weight,
                uint32_t c_in, uint32_t c_out)
{
    check_same_device(*inp, *weight, "conv2d_multi");

    if (inp->shape.size() != 3)
        throw std::invalid_argument("conv2d_multi: input 必须是 3D (C_IN, H, W)");
    if (weight->shape.size() != 4)
        throw std::invalid_argument("conv2d_multi: weight 必须是 4D (C_OUT, C_IN, K, K)");

    uint32_t H  = static_cast<uint32_t>(inp->shape[1]);
    uint32_t W  = static_cast<uint32_t>(inp->shape[2]);
    uint32_t Kh = static_cast<uint32_t>(weight->shape[2]);
    uint32_t Kw = static_cast<uint32_t>(weight->shape[3]);

    if (Kh != Kw)
        throw std::invalid_argument("conv2d_multi: kernel 必须为方形");

    uint32_t out_h = H - Kh + 1, out_w = W - Kw + 1;
    std::vector<ssize_t> out_shape = {static_cast<ssize_t>(c_out),
                                       static_cast<ssize_t>(out_h),
                                       static_cast<ssize_t>(out_w)};
    py::array_t<float> out_arr(out_shape);
    float *out_ptr = static_cast<float *>(out_arr.request().ptr);

    check_error(gpgpuConv2DMulti(inp->device->dev, inp->device->ops.conv2d,
                                  inp->ptr(), weight->ptr(), out_ptr,
                                  H, W, Kh, c_in, c_out),
                "gpgpuConv2DMulti");

    auto result = std::make_shared<TensorWrapper>(out_arr, inp->device);
    result->shape = out_shape;
    return result;
}

// ============================================================
// 模块定义
// ============================================================

PYBIND11_MODULE(gpgpu, m) {
    m.doc() = R"(
GPGPU Python bindings — PyTorch-like API for RISC-V SIMT GPU

Quick start:
    import gpgpu, numpy as np

    dev = gpgpu.Device()

    a = gpgpu.Tensor(np.array([1, 2, 3, 4], dtype=np.float32), dev)
    b = gpgpu.Tensor(np.array([5, 6, 7, 8], dtype=np.float32), dev)

    c = a + b              # VecAdd
    d = gpgpu.relu(a)      # ReLU (返回新 Tensor)
    e = a.reshape([2,2]) @ b.reshape([2,2])  # MatMul

    print(c.numpy())       # 转回 numpy 数组
)";

    // ---- Device ----
    py::class_<DeviceWrapper, std::shared_ptr<DeviceWrapper>>(m, "Device",
        R"(
GPGPU 设备句柄。

构造时自动完成：gpgpuInit -> gpgpuReset -> gpgpuLoadOperators。
可通过环境变量 GPGPU_KERNEL_DIR 指定 kernel 二进制目录（默认 "bin/kernels"）。

示例：
    dev = gpgpu.Device()
        )")
        .def(py::init<>())
        .def("__repr__", [](const DeviceWrapper &) {
            return "<gpgpu.Device>";
        });

    // ---- Tensor ----
    py::class_<TensorWrapper, std::shared_ptr<TensorWrapper>>(m, "Tensor",
        R"(
持有 float32 数据的 GPGPU tensor。

数据存储在 CPU 侧（numpy 数组），每次算子调用时自动完成
Host→Device 拷贝、kernel 执行和 Device→Host 拷贝。

参数：
    data   : numpy.ndarray，将被转换为 float32
    device : gpgpu.Device

示例：
    a = gpgpu.Tensor(np.array([1, 2, 3], dtype=np.float32), dev)
    b = gpgpu.Tensor(np.ones((3, 3)), dev)   # 自动转 float32
        )")
        .def(py::init([](py::array_t<float> arr,
                         std::shared_ptr<DeviceWrapper> dev) {
                return std::make_shared<TensorWrapper>(arr, dev);
             }),
             py::arg("data"), py::arg("device"))
        .def("numpy", &TensorWrapper::numpy,
             "返回底层 numpy 数组（零拷贝共享内存）。")
        .def_property_readonly("shape",
            [](const TensorWrapper &t) { return py::tuple(py::cast(t.shape)); },
            "Tensor 的形状（tuple）。")
        .def_property_readonly("ndim",
            [](const TensorWrapper &t) { return (int)t.shape.size(); },
            "维度数。")
        .def("numel", &TensorWrapper::numel,
             "元素总数。")
        .def("reshape",
            [](std::shared_ptr<TensorWrapper> self,
               std::vector<ssize_t> new_shape) {
                size_t new_n = 1;
                for (auto d : new_shape) {
                    if (d <= 0)
                        throw std::invalid_argument("reshape: 无效维度 " +
                                                     std::to_string(d));
                    new_n *= static_cast<size_t>(d);
                }
                if (new_n != self->numel())
                    throw std::invalid_argument(
                        "reshape: 元素数不匹配（" +
                        std::to_string(self->numel()) + " → " +
                        std::to_string(new_n) + "）");

                py::array_t<float> reshaped = self->data.reshape(new_shape);
                auto result = std::make_shared<TensorWrapper>(reshaped, self->device);
                result->shape = new_shape;
                return result;
            },
            py::arg("shape"),
            "返回指定形状的新 Tensor（共享数据，无拷贝）。")
        .def("__repr__",
            [](const TensorWrapper &t) {
                return "Tensor(shape=" + t.shape_str() + ", dtype=float32)";
            })
        .def("__add__",
            [](std::shared_ptr<TensorWrapper> self,
               std::shared_ptr<TensorWrapper> other) {
                return op_add(self, other);
            })
        .def("__matmul__",
            [](std::shared_ptr<TensorWrapper> self,
               std::shared_ptr<TensorWrapper> other) {
                return op_matmul(self, other);
            });

    // ---- 算子函数 ----

    m.def("add", &op_add, py::arg("a"), py::arg("b"),
          "向量加法：c = a + b（逐元素，shape 必须一致）。");

    m.def("relu", &op_relu, py::arg("x"),
          "ReLU：y = max(0, x)（返回新 Tensor，不修改输入）。");

    m.def("matmul", &op_matmul, py::arg("a"), py::arg("b"),
          "矩阵乘法：C(M,N) = A(M,K) * B(K,N)。两者均需为 2D tensor。");

    m.def("softmax", &op_softmax, py::arg("x"),
          "Softmax：GPU 计算 exp，CPU 归一化。输入必须为 1D tensor。");

    m.def("maxpool2x2", &op_maxpool2x2, py::arg("x"),
          "MaxPool 2×2：输出形状 (H//2, W//2)。输入需为 2D，H 和 W 须为偶数。");

    m.def("conv2d", &op_conv2d, py::arg("input"), py::arg("weight"),
          "单通道 2D 卷积（步长1，无 padding）。\n"
          "input: 2D (H, W)  weight: 2D (K, K)  output: 2D (H-K+1, W-K+1)");

    m.def("conv2d_multi", &op_conv2d_multi,
          py::arg("input"), py::arg("weight"),
          py::arg("c_in"), py::arg("c_out"),
          "多通道 2D 卷积（步长1，无 padding）。\n"
          "input: 3D (C_IN, H, W)  weight: 4D (C_OUT, C_IN, K, K)  "
          "output: 3D (C_OUT, H-K+1, W-K+1)");
}
