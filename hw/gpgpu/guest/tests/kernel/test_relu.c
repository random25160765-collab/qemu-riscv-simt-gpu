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
#define DATA_OFFSET     0x100000

#define N 16  // 测试元素数量

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

int main() {
    printf("=== GPGPU ReLU Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    // 加载 kernel
    size_t ksize;
    load_kernel(vram, "bin/kernels/relu.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    // 准备测试数据
    float data[N];
    srand(time(NULL));
    printf("Input data:\n");
    for (int i = 0; i < N; i++) {
        data[i] = (float)(rand() % 200 - 100) / 10.0f;  // -10.0 ~ 10.0
        printf("%6.2f ", data[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    // CPU 计算期望结果
    float expected[N];
    for (int i = 0; i < N; i++) {
        expected[i] = data[i] > 0 ? data[i] : 0.0f;
    }

    // 写入 VRAM
    memcpy(vram + DATA_OFFSET, data, sizeof(data));

    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {1, 1, 1},
        .block_dim = {N, 1, 1},
    };
    printf("\nLaunching: grid=(%d,1,1), block=(%d,1,1)\n", N, 256);
    
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    // 读取结果
    float result[N];
    memcpy(result, vram + DATA_OFFSET, sizeof(result));

    // 验证
    printf("\nResults:\n");
    printf("Index   Input    GPU      Expected  Match\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        int match = (fabsf(result[i] - expected[i]) < 0.01f);
        printf("%5d: %7.2f %7.2f %7.2f   %s\n", 
               i, data[i], result[i], expected[i], match ? "OK" : "FAIL");
        if (!match) errors++;
    }

    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}