#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "gpgpu_ioctl.h"

#define VRAM_SIZE       (64 * 1024 * 1024)

// VRAM 布局
#define INPUT_OFFSET    0x100000   // 输入图像
#define CONV1_W_OFFSET  0x200000   // Conv1 权重 (3x3)
#define CONV1_OUT       0x300000   // Conv1 输出 (6x6)
#define POOL1_OUT       0x400000   // Pool1 输出 (3x3)
#define CONV2_W_OFFSET  0x200000   // Conv2 权重 (3x3) - 复用
#define CONV2_OUT       0x300000   // Conv2 输出 (1x1) - 复用
#define FC_W_OFFSET     0x500000   // FC 权重 (10x1)
#define FC_OUT          0x600000   // FC 输出 (10)
#define SOFTMAX_OUT     0x700000   // Softmax 输出 (10)

void load_file(void *vram, uint32_t offset, const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { 
        printf("Warning: %s not found, using random weights\n", path);
        *size = 0;
        return;
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *data = malloc(*size);
    fread(data, 1, *size, fp);
    fclose(fp);
    memcpy(vram + offset, data, *size);
    free(data);
}

void run_kernel(int fd, const char *name, uint32_t kernel_addr,
                uint32_t gx, uint32_t gy, uint32_t gz,
                uint32_t bx, uint32_t by, uint32_t bz) {
    printf("  %s... ", name);
    fflush(stdout);
    
    struct gpgpu_kernel_params params = {
        .kernel_addr = kernel_addr,
        .grid_dim = {gx, gy, gz},
        .block_dim = {bx, by, bz},
    };
    
    if (ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params) < 0) {
        perror("ioctl");
        exit(1);
    }
    
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);
    printf("done\n");
}

int main() {
    printf("=== GPGPU Single-Channel CNN ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    // 加载 kernels 到 VRAM
    size_t sz;
    load_file(vram, 0x0000, "bin/kernel/conv2d.bin", &sz);
    load_file(vram, 0x1000, "bin/kernel/relu.bin", &sz);
    load_file(vram, 0x2000, "bin/kernel/maxpool.bin", &sz);
    load_file(vram, 0x3000, "bin/kernel/matmul.bin", &sz);
    load_file(vram, 0x4000, "bin/kernel/softmax_exp.bin", &sz);
    printf("Kernels loaded\n\n");

    // 1. 生成随机输入 8x8
    float input[64];
    srand(time(NULL));
    printf("Input (8x8):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            input[i*8+j] = (float)(rand() % 100) / 100.0f;
            printf("%5.2f ", input[i*8+j]);
        }
        printf("\n");
    }
    memcpy(vram + INPUT_OFFSET, input, sizeof(input));

    // 2. 生成随机权重
    float conv1_w[9], conv2_w[9], fc_w[10];
    for (int i = 0; i < 9; i++) {
        conv1_w[i] = (float)(rand() % 200 - 100) / 100.0f;
        conv2_w[i] = (float)(rand() % 200 - 100) / 100.0f;
    }
    for (int i = 0; i < 10; i++) {
        fc_w[i] = (float)(rand() % 200 - 100) / 100.0f;
    }

    printf("\nConv1 weights (3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%6.2f ", conv1_w[i*3+j]);
        }
        printf("\n");
    }

    // ========== Layer 1: Conv2D ==========
    printf("\n=== Layer 1: Conv2D (8x8 + 3x3 -> 6x6) ===\n");
    memcpy(vram + CONV1_W_OFFSET, conv1_w, sizeof(conv1_w));
    run_kernel(fd, "Conv1", 0x0000, 8, 8, 1, 3, 1, 1);

    printf("\n=== Debug: Conv1 output (6x6) before ReLU ===\n");
    float conv1_out[36];
    memcpy(conv1_out, vram + CONV1_OUT, sizeof(conv1_out));
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%7.2f ", conv1_out[i*6+j]);
        }
        printf("\n");
    }

    // ========== Layer 2: ReLU ==========
    printf("\n=== Layer 2: ReLU ===\n");
    run_kernel(fd, "ReLU1", 0x1000, 36, 1, 1, 256, 1, 1);

    // 在 ReLU1 之后
    printf("\n=== Debug: Conv1 output (6x6) after ReLU ===\n");
    memcpy(conv1_out, vram + CONV1_OUT, sizeof(conv1_out));
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%7.2f ", conv1_out[i*6+j]);
        }
        printf("\n");
    }

    // ========== Layer 3: MaxPool ==========
    printf("\n=== Layer 3: MaxPool (6x6 -> 3x3) ===\n");
    // 把 CONV1_OUT 复制到 INPUT_OFFSET（因为 MaxPool 从 0x100000 读）
    memcpy(vram + INPUT_OFFSET, vram + CONV1_OUT, 36 * sizeof(float));
    run_kernel(fd, "Pool1", 0x2000, 6, 6, 1, 256, 1, 1);
    // 结果在 0x200000 (POOL1_OUT)

    printf("\n=== Debug: Pool1 output (3x3) ===\n");
    float pool1[9];
    memcpy(pool1, vram + POOL1_OUT, sizeof(pool1));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%8.4f ", pool1[i*3+j]);
        }
        printf("\n");
    }

    // ========== Layer 4: Conv2D ==========
    printf("\n=== Layer 4: Conv2D (3x3 + 3x3 -> 1x1) ===\n");
    // 把 POOL1_OUT 复制到 INPUT_OFFSET
    memcpy(vram + INPUT_OFFSET, vram + POOL1_OUT, 9 * sizeof(float));
    memcpy(vram + CONV2_W_OFFSET, conv2_w, sizeof(conv2_w));
    run_kernel(fd, "Conv2", 0x0000, 3, 3, 1, 3, 1, 1);
    // 结果在 0x300000 (CONV2_OUT) - 1x1

    // ========== Layer 5: ReLU ==========
    printf("\n=== Layer 5: ReLU ===\n");
    run_kernel(fd, "ReLU2", 0x1000, 1, 1, 1, 256, 1, 1);

    // 在 Layer 5 之后
    printf("\n=== Debug: Conv2 output (before ReLU) ===\n");
    float conv2_before;
    memcpy(&conv2_before, vram + CONV2_OUT, sizeof(float));
    printf("  value = %.4f\n", conv2_before);

    printf("\n=== Debug: Conv2 output (after ReLU) ===\n");
    float conv2_after;
    memcpy(&conv2_after, vram + CONV2_OUT, sizeof(float));  // ReLU 是原地操作
    printf("  value = %.4f\n", conv2_after);

    // ========== Layer 6: Linear (1 -> 10) ==========
    printf("\n=== Layer 6: Linear (1 -> 10) ===\n");
    // MatMul: A(10x1) * B(1x1) ？不对，Linear 是 W * x
    // 我们把 x 当作 1x1，W 当作 10x1，结果 10x1
    // 参数: M=10, K=1, N=1
    memcpy(vram + 0x100000, fc_w, 10 * sizeof(float));   // A = W (10x1)
    // B = CONV2_OUT 已经在 0x200000？需要复制到 0x200000
    memcpy(vram + 0x200000, vram + CONV2_OUT, sizeof(float));
    run_kernel(fd, "FC", 0x3000, 10, 1, 1, 1, 1, 1);
    // 结果在 0x300000

    printf("\n=== Debug: FC output ===\n");
    float fc_out[10];
    memcpy(fc_out, vram + 0x300000, sizeof(fc_out));
    for (int i = 0; i < 10; i++) {
        printf("  fc_out[%d] = %.4f\n", i, fc_out[i]);
    }

    // ========== Layer 7: Softmax ==========
    printf("\n=== Layer 7: Softmax ===\n");
    // 把 FC 结果复制到 0x100000
    memcpy(vram + 0x100000, vram + 0x300000, 10 * sizeof(float));
    run_kernel(fd, "Softmax", 0x4000, 10, 1, 1, 256, 1, 1);

    // 读取最终结果
    float output[10];
    memcpy(output, vram + 0x200000, sizeof(output));  // Softmax 输出在 0x200000

    printf("\n=== Final Output (Softmax) ===\n");
    float sum = 0;
    int pred = 0;
    for (int i = 0; i < 10; i++) {
        printf("  Class %d: %.4f\n", i, output[i]);
        sum += output[i];
        if (output[i] > output[pred]) pred = i;
    }
    printf("  Sum = %.4f\n", sum);
    printf("\nPredicted class: %d\n", pred);

    munmap(vram, VRAM_SIZE);
    close(fd);
    return 0;
}