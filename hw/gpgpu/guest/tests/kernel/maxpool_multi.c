/*
 * maxpool_multi kernel test — 2x2 maxpool over multi-channel input
 *
 * Input:  [C, H, W] @ 0x100000  (row-major)
 * Output: [C, H/2, W/2] @ 0x200000
 *
 * blockIdx.y = c, blockIdx.x = out_y, threadIdx.x = out_x
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
#define OUT_OFFSET   0x200000
#define C            2
#define H            4
#define W            4
#define OUT_H        (H / 2)
#define OUT_W        (W / 2)
#define N            (C * H * W)
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
    printf("=== maxpool_multi Kernel Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    size_t ksize;
    load_kernel(vram, "bin/kernel/maxpool_multi.bin", &ksize);
    printf("Loaded kernel: %zu bytes\n", ksize);

    float input[N], expected[C * OUT_H * OUT_W];
    srand(time(NULL));

    printf("Input (%dx%dx%d):\n", C, H, W);
    for (int c = 0; c < C; c++) {
        printf("  channel %d:\n", c);
        for (int y = 0; y < H; y++) {
            printf("    ");
            for (int x = 0; x < W; x++) {
                int idx = c * H * W + y * W + x;
                input[idx] = (float)(rand() % 200) / 10.0f;
                printf("%5.1f ", input[idx]);
            }
            printf("\n");
        }
    }

    /* CPU reference: 2x2 maxpool */
    for (int c = 0; c < C; c++) {
        for (int oy = 0; oy < OUT_H; oy++) {
            for (int ox = 0; ox < OUT_W; ox++) {
                float max_val = -INFINITY;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = oy * 2 + dy;
                        int ix = ox * 2 + dx;
                        float v = input[c * H * W + iy * W + ix];
                        if (v > max_val) max_val = v;
                    }
                }
                expected[c * OUT_H * OUT_W + oy * OUT_W + ox] = max_val;
            }
        }
    }

    memcpy(vram + IN_OFFSET, input, sizeof(input));
    memset(vram + OUT_OFFSET, 0, sizeof(expected));

    /* grid: {out_h, C, H} — kernel uses grid_dim[2] for H */
    struct gpgpu_kernel_params params = {
        .kernel_addr = 0,
        .grid_dim = {OUT_H, C, H},
        .block_dim = {W, 1, 1},
    };
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);

    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    float result[C * OUT_H * OUT_W];
    memcpy(result, vram + OUT_OFFSET, sizeof(result));

    int errors = 0;
    printf("\nOutput (%dx%dx%d):\n", C, OUT_H, OUT_W);
    for (int c = 0; c < C; c++) {
        printf("  channel %d:\n", c);
        for (int oy = 0; oy < OUT_H; oy++) {
            printf("    ");
            for (int ox = 0; ox < OUT_W; ox++) {
                int idx = c * OUT_H * OUT_W + oy * OUT_W + ox;
                printf("%5.1f ", result[idx]);
                if (fabsf(result[idx] - expected[idx]) > TOLERANCE) {
                    printf("(exp %.1f) ", expected[idx]);
                    errors++;
                }
            }
            printf("\n");
        }
    }

    printf("  errors=%d / %d\n", errors, C * OUT_H * OUT_W);
    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}
