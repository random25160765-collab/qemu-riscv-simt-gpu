#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "gpgpu_ioctl.h"

#define VRAM_SIZE       (64 * 1024 * 1024)
#define INPUT_OFFSET    0x100000
#define KERNEL_OFFSET   0x200000
#define OUTPUT_OFFSET   0x300000

#define IN_H 5
#define IN_W 5
#define K_SIZE 3

void load_kernel(void *vram, const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); exit(1); }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *kernel = malloc(*size);
    fread(kernel, 1, *size, fp);
    fclose(fp);
    memcpy(vram, kernel, *size);
    free(kernel);
}

void cpu_conv2d(float *input, float *kernel, float *output, int H, int W, int K) {
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            float sum = 0;
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    sum += input[(i + ky) * W + (j + kx)] * kernel[ky * K + kx];
                }
            }
            output[i * out_w + j] = sum;
        }
    }
}

int main() {
    printf("=== GPGPU Conv2D Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernels/conv2d.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    // 准备测试数据
    float input[IN_H * IN_W];
    float kernel[K_SIZE * K_SIZE];
    int out_h = IN_H - K_SIZE + 1;
    int out_w = IN_W - K_SIZE + 1;
    float expected[out_h * out_w];
    float result[out_h * out_w];

    srand(time(NULL));
    printf("Input (%dx%d):\n", IN_H, IN_W);
    for (int i = 0; i < IN_H; i++) {
        for (int j = 0; j < IN_W; j++) {
            input[i * IN_W + j] = (float)(rand() % 100) / 10.0f;
            printf("%6.2f ", input[i * IN_W + j]);
        }
        printf("\n");
    }

    printf("\nKernel (%dx%d):\n", K_SIZE, K_SIZE);
    for (int i = 0; i < K_SIZE; i++) {
        for (int j = 0; j < K_SIZE; j++) {
            kernel[i * K_SIZE + j] = (float)(rand() % 100) / 10.0f;
            printf("%6.2f ", kernel[i * K_SIZE + j]);
        }
        printf("\n");
    }

    // CPU 计算
    cpu_conv2d(input, kernel, expected, IN_H, IN_W, K_SIZE);
    printf("\nExpected output (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            printf("%8.2f ", expected[i * out_w + j]);
        }
        printf("\n");
    }

    // 写入 VRAM
    memcpy(vram + INPUT_OFFSET, input, sizeof(input));
    memcpy(vram + KERNEL_OFFSET, kernel, sizeof(kernel));
    memset(vram + OUTPUT_OFFSET, 0, sizeof(result));

    // 启动 kernel
    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {IN_H, IN_W, 1},
        .block_dim = {K_SIZE, 1, 1},
    };
    printf("\nLaunching: grid=(%d,%d,1), block=(%d,1,1)\n", IN_H, IN_W, K_SIZE);
    
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    // 读取结果
    memcpy(result, vram + OUTPUT_OFFSET, sizeof(result));

    printf("\nGPU output:\n");
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            printf("%8.2f ", result[i * out_w + j]);
        }
        printf("\n");
    }

    // 验证
    printf("\nVerification:\n");
    int errors = 0;
    for (int i = 0; i < out_h * out_w; i++) {
        if (fabsf(result[i] - expected[i]) > 0.1f) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], result[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("  All results match!\n");
        printf("\n=== Test PASSED ===\n");
    } else {
        printf("  %d errors found\n", errors);
        printf("\n=== Test FAILED ===\n");
    }

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}