#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "gpgpu_ioctl.h"

#define VRAM_SIZE       (64 * 1024 * 1024)
#define A_OFFSET        0x100000
#define B_OFFSET        0x200000
#define C_OFFSET        0x300000

#define MAT_M 4
#define MAT_N 4
#define MAT_K 4
#define MAX_SHOW    5

void cpu_matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void load_kernel(void *vram, const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *kernel = malloc(*size);
    fread(kernel, 1, *size, fp);
    fclose(fp);
    memcpy(vram, kernel, *size);
    free(kernel);
}

void print_matrix(const char *name, float *M, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== GPGPU Matrix Multiplication Test ===\n\n");

    // 1. 打开设备
    printf("Step 1: Opening device...\n");
    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // 2. mmap VRAM
    printf("Step 2: Mapping VRAM...\n");
    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // 3. 复位设备
    printf("Step 3: Resetting device...\n");
    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    // 4. 加载 kernel
    printf("Step 4: Loading kernel...\n");
    size_t ksize;
    load_kernel(vram, "bin/kernel/matmul.bin", &ksize);
    printf("  Loaded %zu bytes\n", ksize);

    // 5. 准备测试数据
    printf("Step 5: Preparing test data...\n");
    float A[MAT_M * MAT_K], B[MAT_K * MAT_N];
    float C_cpu[MAT_M * MAT_N], C_gpu[MAT_M * MAT_N];

    srand(time(NULL));
    for (int i = 0; i < MAT_M * MAT_K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < MAT_K * MAT_N; i++) B[i] = (float)(rand() % 100) / 10.0f;

    print_matrix("Matrix A", A, MAT_M, MAT_K);
    printf("\n");
    print_matrix("Matrix B", B, MAT_K, MAT_N);
    printf("\n");

    // CPU 计算验证
    cpu_matmul(A, B, C_cpu, MAT_M, MAT_N, MAT_K);
    print_matrix("Expected C (CPU)", C_cpu, MAT_M, MAT_N);
    printf("\n");

    // 6. 写入数据到 VRAM
    printf("Step 6: Writing data to VRAM...\n");
    memcpy(vram + A_OFFSET, A, sizeof(A));
    memcpy(vram + B_OFFSET, B, sizeof(B));
    memset(vram + C_OFFSET, 0, sizeof(C_gpu));
    printf("  A at 0x%x, B at 0x%x, C at 0x%x\n", A_OFFSET, B_OFFSET, C_OFFSET);

    // 7. 配置并启动 kernel
    printf("Step 7: Launching kernel...\n");
    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {MAT_M, MAT_K, 1},   // grid[0]=M, grid[1]=K
        .block_dim = {MAT_N, 1, 1},      // block[0]=N
        .args_addr = 0,
        .shared_mem = 0
    };
    printf("  Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           params.grid_dim[0], params.grid_dim[1], params.grid_dim[2],
           params.block_dim[0], params.block_dim[1], params.block_dim[2]);

    if (ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params) < 0) {
        perror("ioctl LAUNCH_PARAMS");
        munmap(vram, VRAM_SIZE);
        close(fd);
        return 1;
    }

    // 8. 等待完成
    printf("Step 8: Waiting for kernel...\n");
    __u32 status;
    if (ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status) < 0) {
        perror("ioctl WAIT_KERNEL");
        munmap(vram, VRAM_SIZE);
        close(fd);
        return 1;
    }
    printf("  Kernel finished, status=0x%08x\n", status);

    // 9. 读取结果
    printf("Step 9: Reading results...\n");
    memcpy(C_gpu, vram + C_OFFSET, sizeof(C_gpu));
    print_matrix("GPU Result C", C_gpu, MAT_M, MAT_N);

    // 10. 验证
    printf("\nStep 10: Verification...\n");
    int errors = 0;
    for (int i = 0; i < MAT_M * MAT_N; i++) {
        float diff = C_gpu[i] - C_cpu[i];
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors < MAX_SHOW)
                printf("  Mismatch at [%d]: CPU=%.2f, GPU=%.2f\n", i, C_cpu[i], C_gpu[i]);
            errors++;
        }
    }

    if (errors > MAX_SHOW)
        printf("  ... and %d more errors\n", errors - MAX_SHOW);
    if (errors == 0) {
        printf("  All results match!\n");
        printf("\n=== Test PASSED ===\n");
    } else {
        printf("  %d errors found\n", errors);
            if (errors < MAX_SHOW)
                printf("\n=== Test FAILED ===\n");
    }

    // 清理
    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}