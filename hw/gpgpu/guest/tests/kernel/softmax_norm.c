/*
 * softmax_norm kernel test — GPU-side normalisation pass
 *
 * Two-pass softmax: CPU computes sum of exp values, kernel computes 1/sum
 * via frcp.s and normalises.  The sum is written to VRAM 0x400000.
 *
 * input (exp values) @ 0x200000, sum @ 0x400000, output @ 0x300000
 */
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

#define VRAM_SIZE    (64 * 1024 * 1024)
#define IN_OFFSET    0x100000
#define EXP_OFFSET   0x200000
#define OUT_OFFSET   0x300000
#define SUM_OFFSET   0x400000
#define N            256
#define MAX_SHOW   5
#define TOLERANCE    0.01f

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

int main() {
    printf("=== softmax_norm Kernel Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernel/softmax_norm.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    float input[N], exp_vals[N], expected[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        input[i] = (float)(rand() % 200 - 100) / 10.0f;
        exp_vals[i] = expf(input[i]);
    }

    /* CPU: compute sum, then softmax = exp / sum */
    float sum = 0;
    for (int i = 0; i < N; i++) sum += exp_vals[i];
    for (int i = 0; i < N; i++) expected[i] = exp_vals[i] / sum;

    /* Write exp values and sum to VRAM */
    memcpy(vram + EXP_OFFSET, exp_vals, sizeof(exp_vals));
    *(volatile float *)(vram + SUM_OFFSET) = sum;
    memset(vram + OUT_OFFSET, 0, sizeof(float) * N);

    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {N, 1, 1},
        .block_dim = {256, 1, 1},
    };
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);

    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    float result[N];
    memcpy(result, vram + OUT_OFFSET, sizeof(result));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(result[i] - expected[i]) > TOLERANCE) {
            if (errors < MAX_SHOW)
                printf("  [%3d] exp=%.4f sum=%.4f expected=%.4f got=%.4f  FAIL\n",
                   i, exp_vals[i], sum, expected[i], result[i]);
            errors++;
        }
    }
    if (errors > MAX_SHOW)
        printf("  ... and %d more errors\n", errors - MAX_SHOW);
    printf("  errors=%d / %d\n", errors, N);
    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}
