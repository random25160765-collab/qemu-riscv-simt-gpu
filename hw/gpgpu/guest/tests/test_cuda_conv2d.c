/*
 * CUDA-like API 测试: Conv2D（单通道）
 * out = conv(in, weight)，步长 1，无 padding
 */

#include <stdio.h>
#include <stdlib.h>
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
    printf("=== Test: Conv2D (single channel) ===\n");
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

    int H = 5, W = 5, K = 3;
    int out_h = H - K + 1, out_w = W - K + 1;
    float in[25], weight[9], out[9], expected[9];

    printf("Input (%dx%d):\n", H, W);
    for (int i = 0; i < H; i++) {
        printf("  ");
        for (int j = 0; j < W; j++) {
            in[i*W + j] = random_float();
            printf("%7.2f ", in[i*W + j]);
        }
        printf("\n");
    }

    printf("Kernel (%dx%d):\n", K, K);
    for (int i = 0; i < K; i++) {
        printf("  ");
        for (int j = 0; j < K; j++) {
            weight[i*K + j] = random_float() * 0.5f;
            printf("%7.2f ", weight[i*K + j]);
        }
        printf("\n");
    }

    /* CPU 参考实现 */
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            float sum = 0;
            for (int y = 0; y < K; y++)
                for (int x = 0; x < K; x++)
                    sum += in[(i+y)*W + (j+x)] * weight[y*K + x];
            expected[i*out_w + j] = sum;
        }
    }

    err = gpgpuConv2D(dev, ops.conv2d, in, weight, out, H, W, K);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuConv2D failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    printf("Expected (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        printf("  ");
        for (int j = 0; j < out_w; j++)
            printf("%7.2f ", expected[i*out_w + j]);
        printf("\n");
    }

    printf("GPU out  (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        printf("  ");
        for (int j = 0; j < out_w; j++)
            printf("%7.2f ", out[i*out_w + j]);
        printf("\n");
    }

    int errors = 0;
    for (int i = 0; i < out_h * out_w; i++) {
        if (!float_equal(out[i], expected[i], 0.1f)) {
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
