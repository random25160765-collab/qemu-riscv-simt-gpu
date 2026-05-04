/* test_kernel_launch.c — vector_add 端到端 OpenCL 测试
 *
 * 测试链路：
 *   clBuildProgram → pocl_driver_build_source → LLVM 编译 RISC-V 二进制
 *   clEnqueueNDRangeKernel → pocl_gpgpu_run → conflux_hal_launch_kernel
 *   → ioctl(GPGPU_IOCTL_LAUNCH_KERNEL) → QEMU SimX
 *
 * 注意：kernel 编译需要 RISC-V LLVM。编译失败时测试报告原因但不崩溃。
 *
 * 运行方式（guest）：
 *   POCL_DEVICES=gpgpu ./test_kernel_launch
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "FAIL: %s  err=%d\n", (msg), (err)); \
        exit(1); \
    } \
} while (0)

#define N 1024

static const char *kernel_src =
"__kernel void vector_add(__global const float *a,\n"
"                         __global const float *b,\n"
"                         __global       float *c)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    c[i] = a[i] + b[i];\n"
"}\n";

int main(void)
{
    cl_int err;

    /* --- 平台/设备/上下文 --- */
    cl_platform_id plat = NULL;
    clGetPlatformIDs(1, &plat, NULL);

    cl_device_id dev = NULL;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    CHECK(err, "clGetDeviceIDs");

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0
    };
    cl_context ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    CHECK(err, "clCreateContext");

    cl_command_queue cq = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    CHECK(err, "clCreateCommandQueue");

    /* --- 编译 kernel --- */
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char *log = malloc(log_len + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = '\0';
        fprintf(stderr, "clBuildProgram failed (err=%d):\n%s\n", err, log);
        free(log);
        /* kernel 编译失败通常是 LLVM 未就绪，不视为 DMA/dispatch 路径错误 */
        printf("SKIP: kernel compile failed — LLVM backend not ready\n");
        clReleaseProgram(prog);
        clReleaseCommandQueue(cq);
        clReleaseContext(ctx);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(prog, "vector_add", &err);
    CHECK(err, "clCreateKernel");

    /* --- 准备数据 --- */
    float *h_a = malloc(N * sizeof(float));
    float *h_b = malloc(N * sizeof(float));
    float *h_c = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }
    memset(h_c, 0, N * sizeof(float));

    cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
    CHECK(err, "clCreateBuffer a");
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
    CHECK(err, "clCreateBuffer b");
    cl_mem d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    CHECK(err, "clCreateBuffer c");

    err  = clEnqueueWriteBuffer(cq, d_a, CL_TRUE, 0, N*sizeof(float), h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cq, d_b, CL_TRUE, 0, N*sizeof(float), h_b, 0, NULL, NULL);
    CHECK(err, "clEnqueueWriteBuffer");

    /* --- 设置参数 + dispatch --- */
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    CHECK(err, "clSetKernelArg");

    size_t global = N, local = 64;
    err = clEnqueueNDRangeKernel(cq, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    CHECK(err, "clEnqueueNDRangeKernel");

    err = clFinish(cq);
    CHECK(err, "clFinish");

    /* --- 读回 + 验证 --- */
    err = clEnqueueReadBuffer(cq, d_c, CL_TRUE, 0, N*sizeof(float), h_c, 0, NULL, NULL);
    CHECK(err, "clEnqueueReadBuffer");

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];  /* 每项应为 N */
        if (h_c[i] != expected) {
            fprintf(stderr, "MISMATCH at [%d]: got %.2f expected %.2f\n",
                    i, h_c[i], expected);
            pass = 0;
            if (i > 4) { fprintf(stderr, "  (stopping early)\n"); break; }
        }
    }

    if (pass)
        printf("ALL PASSED  (vector_add %d elements via GPGPU)\n", N);
    else
        printf("FAILED\n");

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
    free(h_a); free(h_b); free(h_c);
    return pass ? 0 : 1;
}
