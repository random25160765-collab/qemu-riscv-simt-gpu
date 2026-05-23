/*
 * CUDA-like API 测试: MatMul
 * C = A * B，M×K × K×N → M×N
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
    printf("=== Test: MatMul ===\n");
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

    int M = 4, N = 4, K = 4;
    float a[16], b[16], c[16], expected[16];

    printf("Matrix A (%dx%d):\n", M, K);
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < K; j++) {
            a[i*K + j] = random_float() * 0.5f;
            printf("%7.2f ", a[i*K + j]);
        }
        printf("\n");
    }

    printf("Matrix B (%dx%d):\n", K, N);
    for (int i = 0; i < K; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) {
            b[i*N + j] = random_float() * 0.5f;
            printf("%7.2f ", b[i*N + j]);
        }
        printf("\n");
    }

    /* CPU 参考实现 */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += a[i*K + k] * b[k*N + j];
            expected[i*N + j] = sum;
        }
    }

    err = gpgpuMatMul(dev, ops.matmul, a, b, c, M, N, K);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuMatMul failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }

    printf("Expected (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++)
            printf("%7.2f ", expected[i*N + j]);
        printf("\n");
    }

    printf("GPU out  (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++)
            printf("%7.2f ", c[i*N + j]);
        printf("\n");
    }

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (!float_equal(c[i], expected[i], 0.1f)) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], c[i]);
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
