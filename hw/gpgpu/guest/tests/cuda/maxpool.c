/*
 * CUDA-like API 测试: MaxPool 2x2
 * 对 H×W 输入做步长为 2 的最大池化
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "gpgpu_runtime.h"

static float random_float() {
    return (float)(rand() % 200 - 100) / 100.0f;
}

static int float_equal(float a, float b, float eps) {
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < eps;
}

int main() {
    printf("=== Test: MaxPool 2x2 ===\n");
    srand(time(NULL));

    GPGPUDevice dev;
    GPGPUOperators ops;

    GPGPUError err = gpgpuInit(&dev);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuInit failed: %s\n", gpgpuGetErrorString(err));
        return 1;
    }
    gpgpuReset(dev);

    err = gpgpuLoadOperators(dev, &ops);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuLoadOperators failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    int H = 4, W = 4;
    float in[16], out[4], expected[4];

    printf("Input (%dx%d):\n", H, W);
    for (int i = 0; i < H; i++) {
        printf("  ");
        for (int j = 0; j < W; j++) {
            in[i*W + j] = random_float();
            printf("%7.2f ", in[i*W + j]);
        }
        printf("\n");
    }

    /* CPU MaxPool 参考实现 */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float max_val = -INFINITY;
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    float val = in[(i*2+y)*W + (j*2+x)];
                    if (val > max_val) max_val = val;
                }
            }
            expected[i*2 + j] = max_val;
        }
    }

    err = gpgpuMaxPool2x2(dev, ops.maxpool, in, out, H, W);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuMaxPool2x2 failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    printf("Expected (%dx%d):\n", H/2, W/2);
    for (int i = 0; i < 2; i++) {
        printf("  ");
        for (int j = 0; j < 2; j++)
            printf("%7.2f ", expected[i*2 + j]);
        printf("\n");
    }

    printf("GPU out  (%dx%d):\n", H/2, W/2);
    for (int i = 0; i < 2; i++) {
        printf("  ");
        for (int j = 0; j < 2; j++)
            printf("%7.2f ", out[i*2 + j]);
        printf("\n");
    }

    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (!float_equal(out[i], expected[i], 0.01f)) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], out[i]);
            errors++;
        }
    }

    gpgpuClose(dev);

    if (errors == 0) {
        printf("[PASS]\n");
        return 0;
    } else {
        printf("[FAIL] %d errors\n", errors);
        return 1;
    }
}
