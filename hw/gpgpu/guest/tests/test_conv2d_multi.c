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

#define IN_H       4
#define IN_W       4
#define C_IN       2
#define K_SIZE     3
#define C_OUT      2

void load_kernel(void *vram, uint32_t offset, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *k = malloc(size);
    fread(k, 1, size, fp);
    fclose(fp);
    memcpy(vram + offset, k, size);
    free(k);
    printf("Loaded %s at 0x%x (%zu bytes)\n", path, offset, size);
}

void print_tensor(const char *name, float *t, int c, int h, int w) {
    printf("%s (%dx%dx%d):\n", name, c, h, w);
    for (int ci = 0; ci < c; ci++) {
        printf("Channel %d:\n", ci);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                printf("%8.3f ", t[ci * h * w + i * w + j]);
            }
            printf("\n");
        }
    }
}

void cpu_conv2d_multi(float *input, float *weight, float *output,
                      int H, int W, int C_in, int K, int C_out) {
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    for (int oc = 0; oc < C_out; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = 0;
                for (int ic = 0; ic < C_in; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int in_idx = ic * H * W + (oy + ky) * W + (ox + kx);
                            int w_idx = oc * C_in * K * K + ic * K * K + ky * K + kx;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                output[oc * out_h * out_w + oy * out_w + ox] = sum;
            }
        }
    }
}

int main() {
    printf("=== GPGPU Multi-Channel Conv2D Test ===\n\n");

    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    void *vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) { perror("mmap"); return 1; }

    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);

    load_kernel(vram, 0x0000, "conv2d.bin");

    int out_h = IN_H - K_SIZE + 1;
    int out_w = IN_W - K_SIZE + 1;
    int in_size = C_IN * IN_H * IN_W;
    int w_size = C_OUT * C_IN * K_SIZE * K_SIZE;
    int out_size = C_OUT * out_h * out_w;
    int total_blocks = C_OUT * out_h;

    float *input = malloc(in_size * sizeof(float));
    float *weight = malloc(w_size * sizeof(float));
    float *expected = malloc(out_size * sizeof(float));
    float *result = malloc(out_size * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < in_size; i++)
        input[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < w_size; i++)
        weight[i] = (float)(rand() % 200 - 100) / 10.0f;

    print_tensor("Input", input, C_IN, IN_H, IN_W);
    print_tensor("Weight", weight, C_OUT * C_IN, K_SIZE, K_SIZE);
    cpu_conv2d_multi(input, weight, expected, IN_H, IN_W, C_IN, K_SIZE, C_OUT);
    print_tensor("Expected", expected, C_OUT, out_h, out_w);

    memcpy(vram + 0x100000, input, in_size * sizeof(float));
    memcpy(vram + 0x200000, weight, w_size * sizeof(float));
    memset(vram + 0x300000, 0, out_size * sizeof(float));

    struct gpgpu_kernel_params params = {
        .kernel_addr = 0x0000,
        .grid_dim = {IN_H, IN_W, C_IN},
        .block_dim = {K_SIZE, C_OUT, 1},
    };

    printf("\nLaunching: H=%d W=%d C_in=%d K=%d C_out=%d\n", IN_H, IN_W, C_IN, K_SIZE, C_OUT);
    printf("Total blocks: %d\n", total_blocks);

    if (ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params) < 0) {
        perror("LAUNCH_PARAMS");
        return 1;
    }

    __u32 status;
    ioctl(fd, GPGPU_IOCTL_WAIT_KERNEL, &status);

    memcpy(result, vram + 0x300000, out_size * sizeof(float));
    print_tensor("GPU Result", result, C_OUT, out_h, out_w);

    int errors = 0;
    for (int i = 0; i < out_size; i++) {
        if (fabsf(result[i] - expected[i]) > 0.1f) {
            errors++;
        }
    }
    printf("\n=== %s ===\n", errors == 0 ? "Test PASSED" : "Test FAILED");

    free(input); free(weight); free(expected); free(result);
    munmap(vram, VRAM_SIZE);
    close(fd);
    return errors ? 1 : 0;
}