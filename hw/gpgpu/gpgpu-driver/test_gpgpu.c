/*
 * GPGPU 完整测试程序（带调试）
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <time.h>
#include "gpgpu_ioctl.h"

#define DEVICE_PATH     "/dev/gpgpu0"
#define VRAM_SIZE       (64 * 1024 * 1024)  // 64MB
#define N               1024                // 向量长度
#define BLOCK_SIZE      256                 // 每个 Block 的线程数

// VRAM 中的偏移量
#define KERNEL_OFFSET   0x00000000  // Kernel 代码
#define VECTOR_A_OFFSET 0x00100000  // 向量 A (1MB)
#define VECTOR_B_OFFSET 0x00200000  // 向量 B (2MB)
#define VECTOR_C_OFFSET 0x00300000  // 向量 C (3MB)

// 辅助函数：读取文件内容
static void *read_file(const char *path, size_t *size)
{
    FILE *fp = fopen(path, "rb");
    void *data;
    
    if (!fp) {
        perror("fopen");
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    data = malloc(*size);
    if (!data) {
        perror("malloc");
        fclose(fp);
        return NULL;
    }
    
    fread(data, 1, *size, fp);
    fclose(fp);
    
    return data;
}

// 等待内核完成（使用 ioctl）
static int wait_kernel_done(int fd)
{
    __u32 status;
    int ret;
    
    printf("  Waiting for kernel...\n");
    fflush(stdout);
    
    ret = ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);
    if (ret < 0) {
        perror("ioctl WAIT_KERNEL");
        return ret;
    }
    
    printf("  Kernel finished, status=0x%08x\n", status);
    return 0;
}

// 打印十六进制数据
static void hex_dump(const char *label, void *data, size_t len)
{
    unsigned char *bytes = (unsigned char *)data;
    printf("%s:\n", label);
    for (size_t i = 0; i < len && i < 64; i++) {
        printf("%02x ", bytes[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (len > 64) printf("... (%zu more bytes)\n", len - 64);
    printf("\n");
}

int main(int argc, char *argv[])
{
    int fd;
    void *vram;
    float *a, *b, *c_cpu, *c_gpu;
    void *kernel_bin;
    size_t kernel_size;
    int i;
    int errors = 0;
    int ret;

    printf("=== GPGPU Full Test (with Debug) ===\n");
    printf("Vector size: %d elements\n", N);
    printf("Block size: %d threads\n", BLOCK_SIZE);
    printf("Grid size: %d blocks\n\n", (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ========== 1. 准备测试数据 ==========
    printf("Preparing test data...\n");
    a = malloc(N * sizeof(float));
    b = malloc(N * sizeof(float));
    c_cpu = malloc(N * sizeof(float));
    c_gpu = malloc(N * sizeof(float));

    srand(time(NULL));
    for (i = 0; i < N; i++) {
        a[i] = (float)(rand() % 1000) / 10.0f;
        b[i] = (float)(rand() % 1000) / 10.0f;
        c_cpu[i] = a[i] + b[i];
    }
    printf("  -> %d elements prepared\n", N);
    printf("  -> A[0]=%.2f, B[0]=%.2f, CPU_C[0]=%.2f\n\n", a[0], b[0], c_cpu[0]);

    // ========== 2. 加载 Kernel 二进制 ==========
    printf("Loading kernel binary...\n");
    kernel_bin = read_file("vector_add.bin", &kernel_size);
    if (!kernel_bin) {
        fprintf(stderr, "Failed to load vector_add.bin\n");
        ret = 1;
        goto err_free_data;
    }
    printf("  -> Loaded %zu bytes\n", kernel_size);
    hex_dump("Kernel binary", kernel_bin, kernel_size);

    // ========== 3. 打开设备 ==========
    printf("Opening device: %s\n", DEVICE_PATH);
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("open failed");
        ret = 1;
        goto err_free_kernel;
    }
    printf("  -> fd = %d\n\n", fd);

    // ========== 4. mmap VRAM ==========
    printf("Mapping VRAM...\n");
    vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) {
        perror("mmap failed");
        ret = 1;
        goto err_close;
    }
    printf("  -> VRAM mapped at %p\n\n", vram);

    // ========== 5. 拷贝数据到 VRAM ==========
    printf("Copying data to VRAM...\n");
    
    // Kernel 代码
    memset(vram + KERNEL_OFFSET, 0, kernel_size);  // 先清零
    memcpy(vram + KERNEL_OFFSET, kernel_bin, kernel_size);
    printf("  -> Kernel loaded at 0x%x\n", KERNEL_OFFSET);
    
    // 输入数据
    memcpy(vram + VECTOR_A_OFFSET, a, N * sizeof(float));
    memcpy(vram + VECTOR_B_OFFSET, b, N * sizeof(float));
    
    // 清零 C 区域
    memset(vram + VECTOR_C_OFFSET, 0, N * sizeof(float));
    
    printf("  -> A at 0x%x, B at 0x%x, C at 0x%x\n", 
           VECTOR_A_OFFSET, VECTOR_B_OFFSET, VECTOR_C_OFFSET);

    // ========== 5.5 验证 VRAM 中的数据 ==========
    printf("\n=== Verifying VRAM before launch ===\n");
    
    // 验证 kernel
    printf("Verifying kernel in VRAM...\n");
    if (memcmp(vram + KERNEL_OFFSET, kernel_bin, kernel_size) == 0) {
        printf("  -> Kernel verified OK!\n");
    } else {
        printf("  -> Kernel MISMATCH!\n");
        hex_dump("Expected", kernel_bin, 32);
        hex_dump("Got", vram + KERNEL_OFFSET, 32);
    }
    
    // 验证 A 和 B
    float *vram_a = (float *)(vram + VECTOR_A_OFFSET);
    float *vram_b = (float *)(vram + VECTOR_B_OFFSET);
    float *vram_c = (float *)(vram + VECTOR_C_OFFSET);
    
    printf("A[0] in VRAM: %.2f (expected %.2f)\n", vram_a[0], a[0]);
    printf("B[0] in VRAM: %.2f (expected %.2f)\n", vram_b[0], b[0]);
    printf("C[0] in VRAM (before): %.2f\n", vram_c[0]);
    
    // ========== 6. 配置并启动内核 ==========
    printf("\nConfiguring and launching kernel...\n");
    
    struct gpgpu_kernel_params params = {
        .grid_dim = {(N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1},
        .block_dim = {BLOCK_SIZE, 1, 1},
        .kernel_addr = KERNEL_OFFSET,
        .args_addr = 0,
        .shared_mem = 0
    };
    
    printf("  Grid: %dx%dx%d\n", 
           params.grid_dim[0], params.grid_dim[1], params.grid_dim[2]);
    printf("  Block: %dx%dx%d\n",
           params.block_dim[0], params.block_dim[1], params.block_dim[2]);
    printf("  Kernel addr: 0x%llx\n", (unsigned long long)params.kernel_addr);
    
    ret = ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    if (ret < 0) {
        perror("ioctl LAUNCH_PARAMS");
        ret = 1;
        goto err_unmap;
    }
    printf("  -> Kernel launched!\n\n");

    // ========== 7. 等待完成 ==========
    ret = wait_kernel_done(fd);
    if (ret < 0) {
        ret = 1;
        goto err_unmap;
    }
    printf("\n");
    
    // 在 wait_kernel_done 之后添加
    printf("=== Scanning VRAM for non-zero values ===\n");
    float *vram_float = (float *)vram;
    int found = 0;

    // 扫描前 4MB 的 VRAM
    for (int i = 0; i < (4 * 1024 * 1024) / 4; i++) {
        if (vram_float[i] != 0.0f) {
            printf("Found non-zero at offset 0x%x: %.4f\n", i * 4, vram_float[i]);
            found++;
            if (found > 20) {
                printf("... (more found)\n");
                break;
            }
        }
    }

    if (found == 0) {
        printf("No non-zero values found in first 4MB of VRAM!\n");
    }

    // ========== 7.5 检查 VRAM 中的 C 是否有变化 ==========
    printf("=== Checking VRAM after kernel ===\n");
    printf("C[0] in VRAM (after): %.2f\n", vram_c[0]);
    printf("C[1] in VRAM (after): %.2f\n", vram_c[1]);
    printf("C[2] in VRAM (after): %.2f\n", vram_c[2]);
    printf("C[3] in VRAM (after): %.2f\n", vram_c[3]);
    
    // 检查整个 C 区域是否非零
    int non_zero = 0;
    for (i = 0; i < N; i++) {
        if (vram_c[i] != 0.0f) {
            non_zero++;
        }
    }
    printf("Non-zero elements in C: %d / %d\n\n", non_zero, N);

    // ========== 8. 读回结果 ==========
    printf("Reading results from VRAM...\n");
    memcpy(c_gpu, vram + VECTOR_C_OFFSET, N * sizeof(float));

    // ========== 9. 验证结果 ==========
    printf("\nVerifying results...\n");
    for (i = 0; i < N; i++) {
        float diff = c_gpu[i] - c_cpu[i];
        if (diff < 0) diff = -diff;
        
        if (diff > 0.001f) {
            printf("  Mismatch at index %d: CPU=%.4f, GPU=%.4f, diff=%.4f\n",
                   i, c_cpu[i], c_gpu[i], diff);
            errors++;
            if (errors > 10) {
                printf("  ... too many errors\n");
                break;
            }
        }
    }

    if (errors == 0) {
        printf("  -> All %d elements verified OK!\n", N);
        printf("\n=== Test PASSED ===\n");
        ret = 0;
    } else {
        printf("  -> %d errors found\n", errors);
        printf("\n=== Test FAILED ===\n");
        ret = 1;
    }

    // ========== 10. 显示前几个结果 ==========
    printf("\nFirst 10 results:\n");
    printf("Index   CPU        GPU        Match\n");
    for (i = 0; i < 10 && i < N; i++) {
        float diff = c_gpu[i] - c_cpu[i];
        if (diff < 0) diff = -diff;
        printf("%5d: %8.2f   %8.2f   %s\n", 
               i, c_cpu[i], c_gpu[i], diff < 0.001f ? "OK" : "FAIL");
    }

err_unmap:
    munmap(vram, VRAM_SIZE);
err_close:
    close(fd);
err_free_kernel:
    free(kernel_bin);
err_free_data:
    free(a); free(b); free(c_cpu); free(c_gpu);

    return ret;
}