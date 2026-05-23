#ifndef GPGPU_RUNTIME_H
#define GPGPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================
 * 类型定义
 * ============================================================ */

typedef void* GPGPUDevice;
typedef void* GPGPUKernel;

typedef enum {
    GPGPU_SUCCESS = 0,
    GPGPU_ERROR_INVALID_DEVICE,
    GPGPU_ERROR_INVALID_VALUE,
    GPGPU_ERROR_LAUNCH_FAILED,
    GPGPU_ERROR_TIMEOUT,
    GPGPU_ERROR_OUT_OF_MEMORY,
} GPGPUError;

typedef enum {
    GPGPU_MEMCPY_HOST_TO_DEVICE = 0,
    GPGPU_MEMCPY_DEVICE_TO_HOST,
    GPGPU_MEMCPY_DEVICE_TO_DEVICE,
} GPGPUMemcpyKind;

/* ============================================================
 * 设备管理
 * ============================================================ */

GPGPUError gpgpuInit(GPGPUDevice *dev);
GPGPUError gpgpuReset(GPGPUDevice dev);
GPGPUError gpgpuClose(GPGPUDevice dev);
GPGPUError gpgpuSynchronize(GPGPUDevice dev);

/* ============================================================
 * 内存管理
 * ============================================================ */

GPGPUError gpgpuMalloc(GPGPUDevice dev, void **ptr, size_t size);
GPGPUError gpgpuFree(GPGPUDevice dev, void *ptr);
GPGPUError gpgpuMemcpy(GPGPUDevice dev, void *dst, const void *src, 
                       size_t size, GPGPUMemcpyKind kind);
GPGPUError gpgpuMemset(GPGPUDevice dev, void *ptr, int value, size_t size);

/* ============================================================
 * Kernel 管理
 * ============================================================ */

GPGPUError gpgpuKernelLoad(GPGPUDevice dev, GPGPUKernel *kernel, 
                           const char *path);
GPGPUError gpgpuKernelLoadFromMemory(GPGPUDevice dev, GPGPUKernel *kernel,
                                     const void *data, size_t size);

/* ============================================================
 * Kernel 启动
 * ============================================================ */

GPGPUError gpgpuLaunchKernel(GPGPUDevice dev, GPGPUKernel kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             uint32_t shared_mem);

/* ============================================================
 * 预定义算子
 * ============================================================ */

typedef struct {
    GPGPUKernel relu;
    GPGPUKernel maxpool;
    GPGPUKernel maxpool_multi;
    GPGPUKernel conv2d;
    GPGPUKernel conv2d_multi;
    GPGPUKernel matmul;
    GPGPUKernel softmax;
    GPGPUKernel vecadd;
    GPGPUKernel bias_add;
} GPGPUOperators;

GPGPUError gpgpuLoadOperators(GPGPUDevice dev, GPGPUOperators *ops);

/* ============================================================
 * 高层算子 API
 * ============================================================ */

// ReLU: y = max(0, x)  (原地)
GPGPUError gpgpuReLU(GPGPUDevice dev, GPGPUKernel kernel, 
                     void *x, uint32_t n);

// MaxPool 2x2: out = maxpool(in)
GPGPUError gpgpuMaxPool2x2(GPGPUDevice dev, GPGPUKernel kernel,
                           const void *in, void *out,
                           uint32_t h, uint32_t w);

// Conv2D 单通道: out = conv(in, weight)
GPGPUError gpgpuConv2D(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *in, const void *weight, void *out,
                       uint32_t h, uint32_t w, uint32_t k);

// Conv2D 多通道: out[c_out x out_h x out_w] = sum_over_c_in(conv(in[c_in], weight[c_out][c_in]))
// kernel 传入 ops->conv2d (单通道 kernel)，内部逐通道调用 GPU 并在 CPU 侧累加
GPGPUError gpgpuConv2DMulti(GPGPUDevice dev, GPGPUKernel conv2d_kernel,
                             const void *in, const void *weight, void *out,
                             uint32_t h, uint32_t w, uint32_t k,
                             uint32_t c_in, uint32_t c_out);

// MatMul: C = A * B
GPGPUError gpgpuMatMul(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *a, const void *b, void *c,
                       uint32_t m, uint32_t n, uint32_t k);

// Softmax (GPU 计算 exp, CPU 归一化)
GPGPUError gpgpuSoftmax(GPGPUDevice dev, GPGPUKernel kernel,
                        const void *in, void *out, uint32_t n);

// VectorAdd: C = A + B
GPGPUError gpgpuVecAdd(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *a, const void *b, void *c, uint32_t n);

// BiasAdd: out[c*HW + i] = in[c*HW + i] + bias[c]
// n = C * H * W, hw = H * W
GPGPUError gpgpuBiasAdd(GPGPUDevice dev, GPGPUKernel kernel,
                        const void *in, const void *bias, void *out,
                        uint32_t n, uint32_t hw);

// MaxPool 2x2 多通道: in[C, H, W] -> out[C, H/2, W/2]
GPGPUError gpgpuMaxPool2x2Multi(GPGPUDevice dev, GPGPUKernel kernel,
                                const void *in, void *out,
                                uint32_t c, uint32_t h, uint32_t w);

/* ============================================================
 * 日志级别控制
 * ============================================================ */

/**
 * 日志级别常量（与 QEMU 侧 GPGPULogLevel 对应）
 */
#define GPGPU_LOG_OFF   0
#define GPGPU_LOG_ERROR 1
#define GPGPU_LOG_INFO  2
#define GPGPU_LOG_DEV   3
#define GPGPU_LOG_CORE  4
#define GPGPU_LOG_INST  5
#define GPGPU_LOG_TRACE 6

/**
 * 日志类别掩码（与 QEMU 侧 GPGPULogCategory 对应）
 */
#define GPGPU_CAT_DEVICE  (1 << 0)
#define GPGPU_CAT_CORE    (1 << 1)
#define GPGPU_CAT_INST    (1 << 2)
#define GPGPU_CAT_DMA     (1 << 3)
#define GPGPU_CAT_INTR    (1 << 4)
#define GPGPU_CAT_ALL     0xFF

/**
 * gpgpuSetLogLevel - 运行时修改 QEMU 模拟器侧的日志输出级别
 * @dev:        设备句柄
 * @level:      日志级别 (GPGPU_LOG_OFF ~ GPGPU_LOG_TRACE)
 * @categories: 类别掩码 (GPGPU_CAT_*，0 表示不修改类别)
 *
 * 通过写 GPGPU_REG_LOG_LEVEL 寄存器立即生效，无需重启 QEMU。
 */
GPGPUError gpgpuSetLogLevel(GPGPUDevice dev, uint32_t level, uint32_t categories);

/* ============================================================
 * 工具函数
 * ============================================================ */

const char* gpgpuGetErrorString(GPGPUError error);

#endif /* GPGPU_RUNTIME_H */