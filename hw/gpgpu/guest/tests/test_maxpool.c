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
#define OUTPUT_OFFSET   0x200000

#define H 4
#define W 4

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

float cpu_maxpool(float *input, int h, int w, int out_y, int out_x) {
    float max_val = -INFINITY;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            int in_y = out_y * 2 + y;
            int in_x = out_x * 2 + x;
            if (in_y < h && in_x < w) {
                float val = input[in_y * w + in_x];
                if (val > max_val) max_val = val;
            }
        }
    }
    return max_val;
}

int main() {
    printf("=== GPGPU MaxPool 2x2 Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernels/maxpool.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    // 准备测试数据
    float input[H * W];
    srand(time(NULL));
    printf("Input (%dx%d):\n", H, W);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            input[i * W + j] = (float)(rand() % 100) / 10.0f;
            printf("%6.2f ", input[i * W + j]);
        }
        printf("\n");
    }

    // CPU 计算
    int out_h = H / 2, out_w = W / 2;
    float expected[out_h * out_w];
    printf("\nExpected output (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            expected[i * out_w + j] = cpu_maxpool(input, H, W, i, j);
            printf("%6.2f ", expected[i * out_w + j]);
        }
        printf("\n");
    }

    // 写入 VRAM
    memcpy(vram + INPUT_OFFSET, input, sizeof(input));
    memset(vram + OUTPUT_OFFSET, 0, out_h * out_w * sizeof(float));

    // 启动 kernel
    int threads_per_block = 256;
    // int num_blocks = (out_h * out_w + threads_per_block - 1) / threads_per_block;
    
    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {H, W, 1},  // 用 grid_dim 传递 H 和 W
        .block_dim = {threads_per_block, 1, 1},
    };
    printf("\nLaunching: grid=(%d,%d,1), block=(%d,1,1)\n", H, W, threads_per_block);
    
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    // 读取结果
    float result[out_h * out_w];
    memcpy(result, vram + OUTPUT_OFFSET, sizeof(result));

    // 验证
    printf("\nGPU output:\n");
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            printf("%6.2f ", result[i * out_w + j]);
        }
        printf("\n");
    }

    printf("\nVerification:\n");
    int errors = 0;
    for (int i = 0; i < out_h * out_w; i++) {
        if (fabsf(result[i] - expected[i]) > 0.01f) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], result[i]);
            errors++;
        }
    }

    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}