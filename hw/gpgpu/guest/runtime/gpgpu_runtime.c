#include "gpgpu_runtime.h"
#include "gpgpu_ioctl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <math.h>

#define VRAM_SIZE       (64 * 1024 * 1024)
#define PARAM_OFFSET    0x800000

typedef struct {
    int fd;
    void *vram;
    uint32_t kernel_base;
} GPGPUDevicePriv;

/* ============================================================
 * 设备管理
 * ============================================================ */

GPGPUError gpgpuInit(GPGPUDevice *dev) {
    GPGPUDevicePriv *p = calloc(1, sizeof(GPGPUDevicePriv));
    if (!p) return GPGPU_ERROR_OUT_OF_MEMORY;
    
    p->fd = open("/dev/gpgpu0", O_RDWR);
    if (p->fd < 0) {
        free(p);
        return GPGPU_ERROR_INVALID_DEVICE;
    }
    
    p->vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, 
                   MAP_SHARED, p->fd, 0);
    if (p->vram == MAP_FAILED) {
        close(p->fd);
        free(p);
        return GPGPU_ERROR_INVALID_DEVICE;
    }
    
    p->kernel_base = 0;
    *dev = p;
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuReset(GPGPUDevice dev) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    ioctl(p->fd, GPGPU_IOCTL_RESET, NULL);
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuClose(GPGPUDevice dev) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    if (p) {
        munmap(p->vram, VRAM_SIZE);
        close(p->fd);
        free(p);
    }
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuSynchronize(GPGPUDevice dev) {
    // 同步操作已在 launch 中完成
    (void)dev;
    return GPGPU_SUCCESS;
}

/* ============================================================
 * 内存管理
 * ============================================================ */

GPGPUError gpgpuMalloc(GPGPUDevice dev, void **ptr, size_t size) {
    // VRAM 静态分配，返回固定偏移
    static uint32_t next_offset = 0x100000;
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    
    if (next_offset + size > VRAM_SIZE) {
        return GPGPU_ERROR_OUT_OF_MEMORY;
    }
    
    *ptr = p->vram + next_offset;
    next_offset += (size + 0xFFF) & ~0xFFF;  // 页对齐
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuFree(GPGPUDevice dev, void *ptr) {
    // 简化：不实现动态释放
    (void)dev; (void)ptr;
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuMemcpy(GPGPUDevice dev, void *dst, const void *src,
                       size_t size, GPGPUMemcpyKind kind) {
    (void)dev;
    switch (kind) {
    case GPGPU_MEMCPY_HOST_TO_DEVICE:
        memcpy(dst, src, size);
        break;
    case GPGPU_MEMCPY_DEVICE_TO_HOST:
        memcpy((void*)src, dst, size);  // 注意参数顺序
        break;
    case GPGPU_MEMCPY_DEVICE_TO_DEVICE:
        memmove(dst, src, size);
        break;
    default:
        return GPGPU_ERROR_INVALID_VALUE;
    }
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuMemset(GPGPUDevice dev, void *ptr, int value, size_t size) {
    memset(ptr, value, size);
    return GPGPU_SUCCESS;
}

/* ============================================================
 * Kernel 管理
 * ============================================================ */

GPGPUError gpgpuKernelLoad(GPGPUDevice dev, GPGPUKernel *kernel,
                           const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return GPGPU_ERROR_INVALID_VALUE;
    
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void *data = malloc(size);
    if (!data) {
        fclose(fp);
        return GPGPU_ERROR_OUT_OF_MEMORY;
    }
    
    size_t read_size = fread(data, 1, size, fp);
    fclose(fp);
    
    if (read_size != size) {
        free(data);
        return GPGPU_ERROR_INVALID_VALUE;
    }
    
    GPGPUError err = gpgpuKernelLoadFromMemory(dev, kernel, data, size);
    free(data);
    return err;
}

GPGPUError gpgpuKernelLoadFromMemory(GPGPUDevice dev, GPGPUKernel *kernel,
                                     const void *data, size_t size) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    uint32_t addr = p->kernel_base;
    
    if (addr + size > 0x100000) {
        return GPGPU_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(p->vram + addr, data, size);
    *kernel = (GPGPUKernel)(uintptr_t)addr;
    p->kernel_base += (size + 0xFFF) & ~0xFFF;
    
    return GPGPU_SUCCESS;
}

/* ============================================================
 * Kernel 启动
 * ============================================================ */

GPGPUError gpgpuLaunchKernel(GPGPUDevice dev, GPGPUKernel kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             uint32_t shared_mem) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    
    struct gpgpu_kernel_params kparams = {
        .kernel_addr = (uint64_t)(uintptr_t)kernel,
        .grid_dim = {grid_x, grid_y, grid_z},
        .block_dim = {block_x, block_y, block_z},
        .shared_mem = shared_mem,
    };
    
    if (ioctl(p->fd, GPGPU_IOCTL_LAUNCH_PARAMS, &kparams) < 0) {
        return GPGPU_ERROR_LAUNCH_FAILED;
    }
    
    __u32 status;
    if (ioctl(p->fd, GPGPU_IOCTL_WAIT_KERNEL, &status) < 0) {
        return GPGPU_ERROR_TIMEOUT;
    }
    
    return GPGPU_SUCCESS;
}

/* ============================================================
 * 预定义算子加载
 * ============================================================ */

GPGPUError gpgpuLoadOperators(GPGPUDevice dev, GPGPUOperators *ops) {
    /* 优先使用环境变量 GPGPU_KERNEL_DIR，默认为 "bin/kernels"（项目根目录下） */
    const char *kdir = getenv("GPGPU_KERNEL_DIR");
    if (!kdir) kdir = "bin/kernels";

    char path[512];
    GPGPUError err;

#define LOAD(field, name) \
    snprintf(path, sizeof(path), "%s/%s", kdir, name); \
    err = gpgpuKernelLoad(dev, &ops->field, path); \
    if (err) return err;

    LOAD(relu,          "relu.bin")
    LOAD(maxpool,       "maxpool.bin")
    LOAD(maxpool_multi, "maxpool_multi.bin")
    LOAD(conv2d,        "conv2d.bin")
    LOAD(conv2d_multi,  "conv2d_multi.bin")
    LOAD(matmul,        "matmul.bin")
    LOAD(softmax,       "softmax_exp.bin")
    LOAD(vecadd,        "vector_add.bin")
    LOAD(bias_add,      "bias_add.bin")

#undef LOAD
    return GPGPU_SUCCESS;
}

/* ============================================================
 * 高层算子 API
 * ============================================================ */

GPGPUError gpgpuReLU(GPGPUDevice dev, GPGPUKernel kernel,
                     void *x, uint32_t n) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;

    memcpy(p->vram + 0x100000, x, n * sizeof(float));

    GPGPUError err = gpgpuLaunchKernel(dev, kernel, n, 1, 1, 256, 1, 1, 0);
    if (err) return err;

    memcpy(x, p->vram + 0x100000, n * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuMaxPool2x2(GPGPUDevice dev, GPGPUKernel kernel,
                           const void *in, void *out,
                           uint32_t h, uint32_t w) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    
    memcpy(p->vram + 0x100000, in, h * w * sizeof(float));
    
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, h, w, 1, 256, 1, 1, 0);
    if (err) return err;
    
    uint32_t out_h = h / 2, out_w = w / 2;
    memcpy(out, p->vram + 0x200000, out_h * out_w * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuConv2D(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *in, const void *weight, void *out,
                       uint32_t h, uint32_t w, uint32_t k) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    
    memcpy(p->vram + 0x100000, in, h * w * sizeof(float));
    memcpy(p->vram + 0x200000, weight, k * k * sizeof(float));
    
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, h, w, 1, k, 1, 1, 0);
    if (err) return err;
    
    uint32_t out_h = h - k + 1, out_w = w - k + 1;
    memcpy(out, p->vram + 0x300000, out_h * out_w * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuConv2DMulti(GPGPUDevice dev, GPGPUKernel conv2d_kernel,
                            const void *in, const void *weight, void *out,
                            uint32_t h, uint32_t w, uint32_t k,
                            uint32_t c_in, uint32_t c_out) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    uint32_t out_h = h - k + 1;
    uint32_t out_w = w - k + 1;
    uint32_t out_elems = out_h * out_w;
    const float *in_f = (const float *)in;
    const float *w_f = (const float *)weight;
    float *out_f = (float *)out;

    float *tmp = malloc(out_elems * sizeof(float));
    if (!tmp) return GPGPU_ERROR_OUT_OF_MEMORY;

    /* 初始化输出为 0 */
    memset(out_f, 0, c_out * out_elems * sizeof(float));

    /*
     * 逐 (oc, ic) 对调用单通道 conv2d kernel，CPU 侧累加结果。
     * 输入布局: [c_in][h][w]
     * 权重布局: [c_out][c_in][k][k]
     * 输出布局: [c_out][out_h][out_w]
     */
    for (uint32_t oc = 0; oc < c_out; oc++) {
        for (uint32_t ic = 0; ic < c_in; ic++) {
            memcpy(p->vram + 0x100000,
                   in_f + ic * h * w,
                   h * w * sizeof(float));
            memcpy(p->vram + 0x200000,
                   w_f + (oc * c_in + ic) * k * k,
                   k * k * sizeof(float));

            GPGPUError err = gpgpuLaunchKernel(dev, conv2d_kernel,
                                               h, w, 1, k, 1, 1, 0);
            if (err) { free(tmp); return err; }

            memcpy(tmp, p->vram + 0x300000, out_elems * sizeof(float));

            float *dst = out_f + oc * out_elems;
            for (uint32_t i = 0; i < out_elems; i++)
                dst[i] += tmp[i];
        }
    }

    free(tmp);
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuMatMul(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *a, const void *b, void *c,
                       uint32_t m, uint32_t n, uint32_t k) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    
    memcpy(p->vram + 0x100000, a, m * k * sizeof(float));
    memcpy(p->vram + 0x200000, b, k * n * sizeof(float));
    
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, m, k, 1, n, 1, 1, 0);
    if (err) return err;
    
    memcpy(c, p->vram + 0x300000, m * n * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuSoftmax(GPGPUDevice dev, GPGPUKernel kernel,
                        const void *in, void *out, uint32_t n) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;

    /* 数值稳定：先减去最大值，避免 exp 溢出/精度损失 */
    const float *in_f = (const float *)in;
    float max_val = in_f[0];
    for (uint32_t i = 1; i < n; i++)
        if (in_f[i] > max_val) max_val = in_f[i];

    float *shifted = (float *)(p->vram + 0x100000);
    for (uint32_t i = 0; i < n; i++)
        shifted[i] = in_f[i] - max_val;

    uint32_t num_blocks = (n + 255) / 256;
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, num_blocks, 1, 1, 256, 1, 1, 0);
    if (err) return err;

    float *exp_out = (float *)(p->vram + 0x200000);
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += exp_out[i];
    for (uint32_t i = 0; i < n; i++) ((float*)out)[i] = exp_out[i] / sum;

    return GPGPU_SUCCESS;
}

GPGPUError gpgpuVecAdd(GPGPUDevice dev, GPGPUKernel kernel,
                       const void *a, const void *b, void *c, uint32_t n) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;

    memcpy(p->vram + 0x100000, a, n * sizeof(float));
    memcpy(p->vram + 0x200000, b, n * sizeof(float));

    uint32_t num_blocks = (n + 255) / 256;
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, num_blocks, 1, 1, 256, 1, 1, 0);
    if (err) return err;

    memcpy(c, p->vram + 0x300000, n * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuBiasAdd(GPGPUDevice dev, GPGPUKernel kernel,
                        const void *in, const void *bias, void *out,
                        uint32_t n, uint32_t hw) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;

    memcpy(p->vram + 0x100000, in,   n * sizeof(float));
    memcpy(p->vram + 0x200000, bias, (n / hw) * sizeof(float));

    /* grid_x = ceil(N/256), grid_y = N, grid_z = HW, block_x = 256 */
    uint32_t num_blocks = (n + 255) / 256;
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, num_blocks, n, hw, 256, 1, 1, 0);
    if (err) return err;

    memcpy(out, p->vram + 0x300000, n * sizeof(float));
    return GPGPU_SUCCESS;
}

GPGPUError gpgpuMaxPool2x2Multi(GPGPUDevice dev, GPGPUKernel kernel,
                                const void *in, void *out,
                                uint32_t c, uint32_t h, uint32_t w) {
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;

    memcpy(p->vram + 0x100000, in, c * h * w * sizeof(float));

    uint32_t out_h = h / 2, out_w = w / 2;
    /* grid_x = out_h, grid_y = C, grid_z = H, block_x = out_w, block_y = W */
    GPGPUError err = gpgpuLaunchKernel(dev, kernel, out_h, c, h, out_w, w, 1, 0);
    if (err) return err;

    memcpy(out, p->vram + 0x200000, c * out_h * out_w * sizeof(float));
    return GPGPU_SUCCESS;
}

/* ============================================================
 * 工具函数
 * ============================================================ */

/* ============================================================
 * 日志级别控制
 * ============================================================ */

GPGPUError gpgpuSetLogLevel(GPGPUDevice dev, uint32_t level, uint32_t categories)
{
    GPGPUDevicePriv *p = (GPGPUDevicePriv *)dev;
    struct gpgpu_log_params params = {
        .level      = level,
        .categories = categories,
    };

    if (!p) return GPGPU_ERROR_INVALID_DEVICE;
    if (level > GPGPU_LOG_TRACE) return GPGPU_ERROR_INVALID_VALUE;

    if (ioctl(p->fd, GPGPU_IOCTL_SET_LOG_LEVEL, &params) < 0)
        return GPGPU_ERROR_INVALID_DEVICE;

    return GPGPU_SUCCESS;
}

const char* gpgpuGetErrorString(GPGPUError error) {
    switch (error) {
    case GPGPU_SUCCESS: return "Success";
    case GPGPU_ERROR_INVALID_DEVICE: return "Invalid device";
    case GPGPU_ERROR_INVALID_VALUE: return "Invalid value";
    case GPGPU_ERROR_LAUNCH_FAILED: return "Kernel launch failed";
    case GPGPU_ERROR_TIMEOUT: return "Kernel timeout";
    case GPGPU_ERROR_OUT_OF_MEMORY: return "Out of memory";
    default: return "Unknown error";
    }
}