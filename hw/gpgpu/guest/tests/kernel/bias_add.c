/*
 * bias_add kernel test — out[i] = in[i] + bias[i / HW]
 *
 * The kernel reads N from grid_dim[1] and HW from grid_dim[2].
 * global_id = blockIdx.x * 256 + threadIdx.x
 * c = global_id / HW
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
#define BIAS_OFFSET  0x200000
#define OUT_OFFSET   0x300000
#define C            3
#define H            4
#define W            4
#define N            (C * H * W)
#define HW           (H * W)
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
    printf("=== bias_add Kernel Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernel/bias_add.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    float input[N], bias[C], expected[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        input[i] = (float)(rand() % 200 - 100) / 10.0f;
    for (int c = 0; c < C; c++)
        bias[c] = (float)(rand() % 100) / 10.0f;

    for (int i = 0; i < N; i++)
        expected[i] = input[i] + bias[i / HW];

    memcpy(vram + IN_OFFSET, input, sizeof(input));
    memcpy(vram + BIAS_OFFSET, bias, sizeof(bias));
    memset(vram + OUT_OFFSET, 0, sizeof(float) * N);

    /* grid_dim[0]=ceil(N/256) for blockIdx, grid_dim[1]=N, grid_dim[2]=HW */
    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {(N + 255) / 256, N, HW},
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
                printf("  [%3d] in=%.4f bias[c=%d]=%.4f expected=%.4f got=%.4f  FAIL\n",
                   i, input[i], i / HW, bias[i / HW], expected[i], result[i]);
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
