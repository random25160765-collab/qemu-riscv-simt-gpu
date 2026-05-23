/*
 * cossin kernel test — dual-output cos & sin
 *
 * The kernel writes cos(x[i]) to 0x200000 and sin(x[i]) to 0x300000.
 * Uses hardware fcos.s + fsin.s instructions.
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
#define COS_OFFSET   0x200000
#define SIN_OFFSET   0x300000
#define N            256
#define MAX_SHOW   5
#define TOLERANCE    1e-5f

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
    printf("=== cossin Kernel Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernel/cossin.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    float input[N], expected_cos[N], expected_sin[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        input[i] = (float)(rand() % 6283) / 1000.0f;  /* 0 .. 2π */
        expected_cos[i] = cosf(input[i]);
        expected_sin[i] = sinf(input[i]);
    }

    memcpy(vram + IN_OFFSET, input, sizeof(input));
    memset(vram + COS_OFFSET, 0, sizeof(float) * N);
    memset(vram + SIN_OFFSET, 0, sizeof(float) * N);

    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {N, 1, 1},
        .block_dim = {256, 1, 1},
    };
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);

    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    float result_cos[N], result_sin[N];
    memcpy(result_cos, vram + COS_OFFSET, sizeof(result_cos));
    memcpy(result_sin, vram + SIN_OFFSET, sizeof(result_sin));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int fail = 0;
        if (fabsf(result_cos[i] - expected_cos[i]) > TOLERANCE) fail = 1;
        if (fabsf(result_sin[i] - expected_sin[i]) > TOLERANCE) fail = 1;
        if (fail) {
            if (errors < MAX_SHOW)
                printf("  [%3d] x=%.4f  cos exp=%.6f got=%.6f  sin exp=%.6f got=%.6f  FAIL\n",
                   i, input[i], expected_cos[i], result_cos[i],
                   expected_sin[i], result_sin[i]);
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
