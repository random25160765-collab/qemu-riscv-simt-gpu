/*
 * CUDA-like API 测试: Conv2D Multi-Channel
 * out[C_OUT][out_h][out_w] = sum_{c_in}(conv(in[c_in], weight[c_out][c_in]))
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
    printf("=== Test: Conv2D Multi-Channel ===\n");
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

    int H = 5, W = 5, K = 3, C_IN = 2, C_OUT = 3;
    int out_h = H - K + 1, out_w = W - K + 1;

    float in[2 * 5 * 5];           /* [C_IN][H][W]          */
    float weight[3 * 2 * 3 * 3];   /* [C_OUT][C_IN][K][K]   */
    float out[3 * 3 * 3];          /* [C_OUT][out_h][out_w] */
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
                            int in_idx = ic * H * W + (oy + ky) * W + (ox + kx);
                            int w_idx  = (oc * C_IN + ic) * K * K + ky * K + kx;
                            sum += in[in_idx] * weight[w_idx];
                        }
                    }
                }
                expected[oc * out_h * out_w + oy * out_w + ox] = sum;
            }
        }
    }

    err = gpgpuConv2DMulti(dev, ops.conv2d, in, weight, out, H, W, K, C_IN, C_OUT);
    if (err != GPGPU_SUCCESS) {
        printf("gpgpuConv2DMulti failed: %s\n", gpgpuGetErrorString(err));
        gpgpuClose(dev);
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

    gpgpuClose(dev);

    if (errors == 0) {
        printf("[PASS]\n");
        return 0;
    } else {
        printf("[FAIL] %d errors\n", errors);
        return 1;
    }
}
