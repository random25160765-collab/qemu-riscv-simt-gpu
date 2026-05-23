/*
 * CUDA-like API 测试: Softmax
 * GPU 计算 exp，CPU 归一化
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "gpgpu_runtime.h"

static int float_equal(float a, float b, float eps) {
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < eps;
}

int main() {
    printf("=== Test: Softmax ===\n");
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

    int N = 8;
    float in[8], out[8], expected[8];

    printf("Input: ");
    for (int i = 0; i < N; i++) {
        in[i] = (float)(rand() % 100 - 30) / 10.0f;
        printf("%.2f ", in[i]);
    }
    printf("\n");

    /* CPU 参考实现（数值稳定版） */
    float max_val = in[0];
    for (int i = 1; i < N; i++)
        if (in[i] > max_val) max_val = in[i];
    float sum = 0;
    for (int i = 0; i < N; i++)
        sum += expf(in[i] - max_val);
    for (int i = 0; i < N; i++)
        expected[i] = expf(in[i] - max_val) / sum;

    err = gpgpuSoftmax(dev, ops.softmax, in, out, N);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuSoftmax failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    printf("Expected: [");
    for (int i = 0; i < N; i++) printf("%.4f ", expected[i]);
    printf("]\n");

    printf("GPU out:  [");
    for (int i = 0; i < N; i++) printf("%.4f ", out[i]);
    printf("]\n");

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (!float_equal(out[i], expected[i], 0.05f)) {
            printf("  Mismatch at %d: expected %.4f, got %.4f\n", i, expected[i], out[i]);
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
