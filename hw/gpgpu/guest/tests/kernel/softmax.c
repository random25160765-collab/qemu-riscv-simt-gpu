/*
 * softmax kernel test — full softmax on GPU (self-contained, O(N²))
 *
 * The kernel computes: softmax(x_i) = e^{x_i} / Σ_j e^{x_j}
 * entirely on GPU using fexp.s + fdiv.s. No CPU-side normalisation.
 *
 * N=8 keeps the O(N²) cost manageable for a simulator.
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

#define VRAM_SIZE   (64 * 1024 * 1024)
#define IN_OFFSET   0x100000
#define OUT_OFFSET  0x200000
#define N           8
#define MAX_SHOW   5
#define TOLERANCE   0.01f

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
    printf("=== softmax Kernel Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernel/softmax.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    float input[N], expected[N];
    srand(time(NULL));
    printf("Input:  ");
    for (int i = 0; i < N; i++) {
        input[i] = (float)(rand() % 100 - 30) / 10.0f;
        printf("%.2f ", input[i]);
    }
    printf("\n");

    /* CPU reference: subtract max for numeric stability */
    float max_val = input[0];
    for (int i = 1; i < N; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0;
    for (int i = 0; i < N; i++) sum += expf(input[i] - max_val);
    for (int i = 0; i < N; i++) expected[i] = expf(input[i] - max_val) / sum;

    memcpy(vram + IN_OFFSET, input, sizeof(input));
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

    printf("GPU:    ");
    for (int i = 0; i < N; i++) printf("%.4f ", result[i]);
    printf("\nCPU:    ");
    for (int i = 0; i < N; i++) printf("%.4f ", expected[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(result[i] - expected[i]) > TOLERANCE) {
            if (errors < MAX_SHOW)
                printf("  [%d] x=%.2f expected=%.4f got=%.4f  FAIL\n",
                   i, input[i], expected[i], result[i]);
            errors++;
        }
    }
    if (errors > MAX_SHOW)
        printf("  ... and %d more errors\n", errors - MAX_SHOW);
    printf("\n  errors=%d / %d\n", errors, N);
    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}
