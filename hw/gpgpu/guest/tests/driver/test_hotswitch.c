/*
 * 后端热切换测试
 *
 * 用同一份 vector_add kernel 和同一组数据，先在 built-in 后端跑一遍，
 * 再热切换到 SimX 后端跑一遍，最后比较两次结果是否一致且正确。
 *
 * kernel 内存布局（与 vector_add.S 一致）：
 *   VRAM+0x000000  kernel 代码
 *   VRAM+0x100000  A[]
 *   VRAM+0x200000  B[]
 *   VRAM+0x300000  C[]  (结果)
 *
 * 测试数据：A[i]=i+1, B[i]=10-i  →  C[i]=11.0 (对所有 i 均成立)
 *
 * 编译：由 test.mk 自动处理，运行 `make test-driver-hotswitch`
 * 环境变量：GPGPU_DEVICE、GPGPU_KERNEL_DIR（由顶层 Makefile 导出）
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "gpgpu_ioctl.h"

#define VRAM_SIZE       (64 * 1024 * 1024)
#define N               4   /* 测试元素数 = block_dim.x */

#define KERNEL_OFFSET   0x010000
#define A_OFFSET        0x100000
#define B_OFFSET        0x200000
#define C_OFFSET        0x300000

static int load_kernel(void *vram, const char *kernel_dir, __u32 backend)
{
    char path[256];
    if (backend == GPGPU_BACKEND_SIMX)
        snprintf(path, sizeof(path), 
                 "/mnt/hostshare/kernel/simx/vector_add_simx.bin");
    else
        snprintf(path, sizeof(path), "%s/vector_add.bin", kernel_dir);

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void *buf = malloc(size);
    if (!buf) {
        fclose(fp);
        return -1;
    }

    fread(buf, 1, size, fp);
    fclose(fp);
    memcpy((char *)vram + KERNEL_OFFSET, buf, size);
    free(buf);
    return 0;
}

static void prepare_data(void *vram)
{
    float *A = (float *)((char *)vram + A_OFFSET);
    float *B = (float *)((char *)vram + B_OFFSET);
    float *C = (float *)((char *)vram + C_OFFSET);

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i + 1);   /* 1.0 2.0 3.0 4.0 */
        B[i] = (float)(10 - i);  /* 10.0 9.0 8.0 7.0 */
        C[i] = 0.0f;
    }
}

/* 切换后端、复位、上传数据、执行 kernel、等待完成，结果写入 out[] */
static int run_one(int fd, void *vram, const char *kernel_dir,
                   __u32 backend, float *out)
{
    /* 1. 热切换后端 */
    struct gpgpu_backend_params bp = { .backend = backend };
    if (ioctl(fd, GPGPU_IOCTL_SET_BACKEND, &bp) < 0) {
        perror("GPGPU_IOCTL_SET_BACKEND");
        return -1;
    }

    /* 2. 复位设备，等待就绪 */
    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(20000);

    /* 3. 上传 kernel 代码和输入数据 */
    if (load_kernel(vram, kernel_dir, backend) < 0)
        return -1;
    prepare_data(vram);

    /* 4. 启动 kernel */
    struct gpgpu_kernel_params kp = {
        .grid_dim    = {1, 1, 1},
        .block_dim   = {N, 1, 1},
        .kernel_addr = KERNEL_OFFSET,
        .args_addr   = 0,
        .shared_mem  = 0,
    };
    if (ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &kp) < 0) {
        perror("GPGPU_IOCTL_LAUNCH_PARAMS");
        return -1;
    }

    /* 5. 等待 kernel 完成 */
    __u32 status;
    if (ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status) < 0) {
        perror("GPGPU_IOCTL_WAIT_KERNEL");
        return -1;
    }

    /* 6. 读取结果 */
    memcpy(out, (char *)vram + C_OFFSET, N * sizeof(float));
    return 0;
}

int main(void)
{
    const char *device_path = getenv("GPGPU_DEVICE");
    const char *kernel_dir  = getenv("GPGPU_KERNEL_DIR");

    if (!device_path || !kernel_dir) {
        fprintf(stderr, "Error: GPGPU_DEVICE and GPGPU_KERNEL_DIR must be set\n");
        return 1;
    }

    printf("=== GPGPU Backend Hot-switch Test ===\n");
    printf("device: %s\n", device_path);
    printf("kernel: %s/vector_add.bin\n\n", kernel_dir);

    int fd = open(device_path, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    float result_builtin[N];
    float result_simx[N];

    /* ---- Round 1: Built-in ---- */
    printf("[1/2] Built-in backend\n");
    if (run_one(fd, vram, kernel_dir, GPGPU_BACKEND_BUILTIN, result_builtin) < 0) {
        fprintf(stderr, "  FAILED\n");
        goto err;
    }
    printf("  C[]: ");
    for (int i = 0; i < N; i++)
        printf("%.1f ", result_builtin[i]);
    printf("\n\n");

    /* ---- Round 2: SimX ---- */
    printf("[2/2] SimX backend\n");
    if (run_one(fd, vram, kernel_dir, GPGPU_BACKEND_SIMX, result_simx) < 0) {
        fprintf(stderr, "  FAILED\n");
        goto err;
    }
    printf("  C[]: ");
    for (int i = 0; i < N; i++)
        printf("%.1f ", result_simx[i]);
    printf("\n\n");

    /* ---- 对比验证 ---- */
    printf("%-6s  %-10s  %-10s  %-10s  %s\n",
           "i", "builtin", "simx", "expected", "");
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1) + (float)(10 - i); /* 始终 = 11.0 */
        int ok = (result_builtin[i] == expected) &&
                 (result_simx[i]    == expected);
        printf("  [%d]  %-10.1f  %-10.1f  %-10.1f  %s\n",
               i, result_builtin[i], result_simx[i], expected,
               ok ? "OK" : "FAIL");
        if (!ok)
            pass = 0;
    }

    munmap(vram, VRAM_SIZE);
    close(fd);

    printf("\n=== Test %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;

err:
    munmap(vram, VRAM_SIZE);
    close(fd);
    return 1;
}
