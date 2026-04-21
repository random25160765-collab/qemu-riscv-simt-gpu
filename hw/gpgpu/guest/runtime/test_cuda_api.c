/*
 * GPGPU CUDA-like API 测试程序
 * 编译: gcc -o test_cuda_api test_cuda_api.c gpgpu_runtime.c -Wall -O2 -lm
 * 运行: sudo ./test_cuda_api
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "gpgpu_runtime.h"

#define PRINT_PASS() printf("  [PASS]\n")
#define PRINT_FAIL() printf("  [FAIL]\n")

/* 辅助函数 */
static float random_float() {
    return (float)(rand() % 200 - 100) / 100.0f;
}

static int float_equal(float a, float b, float eps) {
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < eps;
}

static void print_vector(const char *name, float *v, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < 8; i++) {
        printf("%7.2f", v[i]);
        if (i < n - 1 && i < 7) printf(", ");
    }
    if (n > 8) printf(", ...");
    printf("]\n");
}

/* ============================================================
 * 测试用例
 * ============================================================ */

// 测试 1: VectorAdd
int test_vector_add(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 1: VectorAdd ===\n");
    
    int N = 16;
    float a[16], b[16], c[16], expected[16];
    
    printf("Input A: ");
    for (int i = 0; i < N; i++) {
        a[i] = random_float();
        printf("%.2f ", a[i]);
    }
    printf("\n");
    
    printf("Input B: ");
    for (int i = 0; i < N; i++) {
        b[i] = random_float();
        printf("%.2f ", b[i]);
    }
    printf("\n");
    
    for (int i = 0; i < N; i++) {
        expected[i] = a[i] + b[i];
    }
    
    GPGPUError err = gpgpuVecAdd(dev, ops->vecadd, a, b, c, N);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuVecAdd failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }
    
    printf("GPU output: ");
    for (int i = 0; i < N; i++) printf("%.2f ", c[i]);
    printf("\n");
    
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (!float_equal(c[i], expected[i], 0.01f)) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], c[i]);
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 2: ReLU
int test_relu(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 2: ReLU ===\n");
    
    int N = 16;
    float x[16], result[16], expected[16];
    
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        x[i] = random_float();
        printf("%.2f ", x[i]);
    }
    printf("\n");
    
    for (int i = 0; i < N; i++) {
        expected[i] = x[i] > 0 ? x[i] : 0.0f;
    }
    
    memcpy(result, x, sizeof(x));
    GPGPUError err = gpgpuReLU(dev, ops->relu, result, N);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuReLU failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }
    
    printf("Output: ");
    for (int i = 0; i < N; i++) printf("%.2f ", result[i]);
    printf("\n");
    
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (!float_equal(result[i], expected[i], 0.01f)) {
            printf("  Mismatch at %d: expected %.2f, got %.2f\n", i, expected[i], result[i]);
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 3: MaxPool 2x2
int test_maxpool(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 3: MaxPool 2x2 ===\n");
    
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
    
    // CPU MaxPool
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
    
    GPGPUError err = gpgpuMaxPool2x2(dev, ops->maxpool, in, out, H, W);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuMaxPool2x2 failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }
    
    printf("Expected (%dx%d):\n", H/2, W/2);
    for (int i = 0; i < 2; i++) {
        printf("  ");
        for (int j = 0; j < 2; j++) {
            printf("%7.2f ", expected[i*2 + j]);
        }
        printf("\n");
    }
    
    printf("GPU output (%dx%d):\n", H/2, W/2);
    for (int i = 0; i < 2; i++) {
        printf("  ");
        for (int j = 0; j < 2; j++) {
            printf("%7.2f ", out[i*2 + j]);
        }
        printf("\n");
    }
    
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (!float_equal(out[i], expected[i], 0.01f)) {
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 4: Conv2D
int test_conv2d(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 4: Conv2D ===\n");
    
    int H = 5, W = 5, K = 3;
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
    
    // CPU Conv2D
    int out_h = H - K + 1, out_w = W - K + 1;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            float sum = 0;
            for (int y = 0; y < K; y++) {
                for (int x = 0; x < K; x++) {
                    sum += in[(i+y)*W + (j+x)] * weight[y*K + x];
                }
            }
            expected[i*out_w + j] = sum;
        }
    }
    
    GPGPUError err = gpgpuConv2D(dev, ops->conv2d, in, weight, out, H, W, K);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuConv2D failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }
    
    printf("Expected (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        printf("  ");
        for (int j = 0; j < out_w; j++) {
            printf("%7.2f ", expected[i*out_w + j]);
        }
        printf("\n");
    }
    
    printf("GPU output (%dx%d):\n", out_h, out_w);
    for (int i = 0; i < out_h; i++) {
        printf("  ");
        for (int j = 0; j < out_w; j++) {
            printf("%7.2f ", out[i*out_w + j]);
        }
        printf("\n");
    }
    
    int errors = 0;
    for (int i = 0; i < out_h * out_w; i++) {
        if (!float_equal(out[i], expected[i], 0.1f)) {
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 5: MatMul
int test_matmul(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 5: MatMul ===\n");
    
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
    
    // CPU MatMul
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[i*K + k] * b[k*N + j];
            }
            expected[i*N + j] = sum;
        }
    }
    
    GPGPUError err = gpgpuMatMul(dev, ops->matmul, a, b, c, M, N, K);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuMatMul failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }
    
    printf("Expected (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) {
            printf("%7.2f ", expected[i*N + j]);
        }
        printf("\n");
    }
    
    printf("GPU output (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) {
            printf("%7.2f ", c[i*N + j]);
        }
        printf("\n");
    }
    
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (!float_equal(c[i], expected[i], 0.1f)) {
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 6: 多通道 Conv2D
int test_conv2d_multi(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 6: Conv2D Multi-Channel ===\n");

    int H = 5, W = 5, K = 3, C_IN = 2, C_OUT = 3;
    int out_h = H - K + 1, out_w = W - K + 1;

    float in[2 * 5 * 5];        /* [C_IN][H][W]          */
    float weight[3 * 2 * 3 * 3]; /* [C_OUT][C_IN][K][K]   */
    float out[3 * 3 * 3];        /* [C_OUT][out_h][out_w] */
    float expected[3 * 3 * 3];

    for (int i = 0; i < C_IN * H * W; i++)
        in[i] = random_float();
    for (int i = 0; i < C_OUT * C_IN * K * K; i++)
        weight[i] = random_float() * 0.5f;

    /* CPU 参考实现 */
    memset(expected, 0, sizeof(expected));
    for (int oc = 0; oc < C_OUT; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < C_IN; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int in_idx  = ic * H * W + (oy + ky) * W + (ox + kx);
                            int w_idx   = (oc * C_IN + ic) * K * K + ky * K + kx;
                            sum += in[in_idx] * weight[w_idx];
                        }
                    }
                }
                expected[oc * out_h * out_w + oy * out_w + ox] = sum;
            }
        }
    }

    GPGPUError err = gpgpuConv2DMulti(dev, ops->conv2d,
                                      in, weight, out,
                                      H, W, K, C_IN, C_OUT);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuConv2DMulti failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
        return 1;
    }

    int errors = 0;
    for (int oc = 0; oc < C_OUT; oc++) {
        printf("  Channel %d: expected=[", oc);
        for (int i = 0; i < out_h * out_w; i++)
            printf("%6.2f", expected[oc * out_h * out_w + i]);
        printf("] got=[");
        for (int i = 0; i < out_h * out_w; i++)
            printf("%6.2f", out[oc * out_h * out_w + i]);
        printf("]\n");

        for (int i = 0; i < out_h * out_w; i++) {
            if (!float_equal(out[oc * out_h * out_w + i],
                             expected[oc * out_h * out_w + i], 0.1f)) {
                printf("  Mismatch at [oc=%d, i=%d]: expected %.3f, got %.3f\n",
                       oc, i,
                       expected[oc * out_h * out_w + i],
                       out[oc * out_h * out_w + i]);
                errors++;
            }
        }
    }

    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

// 测试 7: Softmax
int test_softmax(GPGPUDevice dev, GPGPUOperators *ops) {
    printf("\n=== Test 6: Softmax ===\n");
    
    int N = 8;
    float in[8], out[8], expected[8];
    
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        in[i] = (float)(rand() % 100 - 30) / 10.0f;
        printf("%.2f ", in[i]);
    }
    printf("\n");
    
    // CPU Softmax
    float max_val = in[0];
    for (int i = 1; i < N; i++) if (in[i] > max_val) max_val = in[i];
    float sum = 0;
    for (int i = 0; i < N; i++) sum += expf(in[i] - max_val);
    for (int i = 0; i < N; i++) expected[i] = expf(in[i] - max_val) / sum;
    
    GPGPUError err = gpgpuSoftmax(dev, ops->softmax, in, out, N);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuSoftmax failed: %s\n", gpgpuGetErrorString(err));
        PRINT_FAIL();
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
            errors++;
        }
    }
    
    if (errors == 0) PRINT_PASS(); else PRINT_FAIL();
    return errors;
}

/* ============================================================
 * 主函数
 * ============================================================ */

int main() {
    printf("============================================================\n");
    printf("        GPGPU CUDA-like API Test Suite\n");
    printf("============================================================\n");
    
    srand(time(NULL));
    
    GPGPUDevice dev;
    GPGPUOperators ops;
    
    printf("\n--- Initializing device ---\n");
    GPGPUError err = gpgpuInit(&dev);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuInit failed: %s\n", gpgpuGetErrorString(err));
        return 1;
    }
    printf("  Device initialized\n");
    
    gpgpuReset(dev);
    printf("  Device reset\n");
    
    printf("\n--- Loading operators ---\n");
    err = gpgpuLoadOperators(dev, &ops);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuLoadOperators failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
        return 1;
    }
    printf("  Operators loaded\n");
    
    int total_errors = 0;
    
    total_errors += test_vector_add(dev, &ops);
    total_errors += test_relu(dev, &ops);
    total_errors += test_maxpool(dev, &ops);
    total_errors += test_conv2d(dev, &ops);
    total_errors += test_matmul(dev, &ops);
    total_errors += test_conv2d_multi(dev, &ops);
    total_errors += test_softmax(dev, &ops);
    
    printf("\n--- Cleaning up ---\n");
    gpgpuClose(dev);
    printf("  Device closed\n");
    
    printf("\n============================================================\n");
    printf("        Test Summary: %d total errors\n", total_errors);
    printf("        %s\n", total_errors == 0 ? "ALL TESTS PASSED!" : "SOME TESTS FAILED");
    printf("============================================================\n");
    
    return total_errors ? 1 : 0;
}