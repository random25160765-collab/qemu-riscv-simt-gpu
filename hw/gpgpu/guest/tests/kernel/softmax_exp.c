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

#define SOFTMAX_N 8
#define MAX_SHOW      5

void load_kernel(void *vram, const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); exit(1); }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *k = malloc(*size);
    fread(k, 1, *size, fp);
    fclose(fp);
    memcpy(vram, k, *size);
    free(k);
}

void run_kernel(int fd, void *vram, const char *kernel_file, int num_elements) {
    size_t ksize;
    load_kernel(vram, kernel_file, &ksize);

    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {num_elements, 1, 1},
        .block_dim = {256, 1, 1},
    };
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);
}

float taylor_exp(float x) {
    float x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    return 1.0f + x + x2/2.0f + x3/6.0f + x4/24.0f;
}

float cpu_exp(float x) {
    if (x < 0) return 1.0f / taylor_exp(-x);
    return taylor_exp(x);
}

int main() {
    printf("=== GPGPU Softmax Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    float input[SOFTMAX_N], gpu_exp[SOFTMAX_N], gpu_softmax[SOFTMAX_N], expected[SOFTMAX_N];

    srand(time(NULL));
    printf("Input data:\n");
    for (int i = 0; i < SOFTMAX_N; i++) {
        input[i] = (float)(rand() % 100 - 30) / 10.0f;
        printf("%6.2f ", input[i]);
    }
    printf("\n");

    // CPU 计算期望值
    float max_val = input[0];
    for (int i = 1; i < SOFTMAX_N; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0;
    for (int i = 0; i < SOFTMAX_N; i++) sum += expf(input[i] - max_val);
    for (int i = 0; i < SOFTMAX_N; i++) expected[i] = expf(input[i] - max_val) / sum;

    // 写入输入
    memcpy(vram + INPUT_OFFSET, input, sizeof(input));
    memset(vram + OUTPUT_OFFSET, 0, sizeof(gpu_exp));

    // 启动 GPU kernel 计算 exp
    printf("\nLaunching GPU exp kernel...\n");
    run_kernel(fd, vram, "bin/kernel/softmax_exp.bin", SOFTMAX_N);

    // 读取 exp 结果
    memcpy(gpu_exp, vram + OUTPUT_OFFSET, sizeof(gpu_exp));

    // CPU 归一化
    float gpu_sum = 0;
    for (int i = 0; i < SOFTMAX_N; i++) gpu_sum += gpu_exp[i];
    for (int i = 0; i < SOFTMAX_N; i++) gpu_softmax[i] = gpu_exp[i] / gpu_sum;

    // 验证
    printf("\nResults:\n");
    int errors = 0;
    for (int i = 0; i < SOFTMAX_N; i++) {
        int match = (fabsf(gpu_softmax[i] - expected[i]) < 0.01f);
        if (!match) {
            if (errors < MAX_SHOW)
                printf("%5d: %7.2f %9.4f %11.4f %9.4f   FAIL\n",
                       i, input[i], gpu_exp[i], gpu_softmax[i], expected[i]);
            errors++;
        }
    }
    if (errors > MAX_SHOW)
        printf("  ... and %d more errors\n", errors - MAX_SHOW);

    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}