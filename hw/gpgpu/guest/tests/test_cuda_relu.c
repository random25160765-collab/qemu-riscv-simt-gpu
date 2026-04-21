/*
 * CUDA-like API 测试: ReLU
 * y = max(0, x)，原地修改
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    printf("=== Test: ReLU ===\n");
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

    int N = 16;
    float x[16], result[16], expected[16];

    printf("Input: ");
    for (int i = 0; i < N; i++) {
        x[i] = random_float();
        printf("%.2f ", x[i]);
    }
    printf("\n");

    for (int i = 0; i < N; i++)
        expected[i] = x[i] > 0 ? x[i] : 0.0f;

    memcpy(result, x, sizeof(x));
    err = gpgpuReLU(dev, ops.relu, result, N);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuReLU failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    printf("Expected: ");
    for (int i = 0; i < N; i++) printf("%.2f ", expected[i]);
    printf("\n");

    printf("GPU out:  ");
    for (int i = 0; i < N; i++) printf("%.2f ", result[i]);
    printf("\n");

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (!float_equal(result[i], expected[i], 0.01f)) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], result[i]);
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
