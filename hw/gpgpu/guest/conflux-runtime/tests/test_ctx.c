/* test_ctx.c — 诊断 clCreateContext -34 问题 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>

int main(void) {
    cl_platform_id plat = NULL;
    cl_uint n = 0;
    clGetPlatformIDs(1, &plat, &n);
    printf("num_platforms=%u  platform=%p\n", n, (void*)plat);

    cl_device_id dev = NULL;
    cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    printf("clGetDeviceIDs err=%d  device=%p\n", err, (void*)dev);

    /* 试1：用 props 显式传 platform */
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0
    };
    cl_context ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    printf("clCreateContext(props) err=%d  ctx=%p\n", err, (void*)ctx);

    /* 试2：NULL props */
    cl_context ctx2 = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    printf("clCreateContext(NULL)  err=%d  ctx=%p\n", err, (void*)ctx2);

    /* 试3：FromType */
    cl_context ctx3 = clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR,
                                              NULL, NULL, &err);
    printf("clCreateContextFromType err=%d  ctx=%p\n", err, (void*)ctx3);

    /* 试4：FromType + props */
    cl_context ctx4 = clCreateContextFromType(props, CL_DEVICE_TYPE_ACCELERATOR,
                                              NULL, NULL, &err);
    printf("clCreateContextFromType(props) err=%d  ctx=%p\n", err, (void*)ctx4);

    if (ctx)  { clReleaseContext(ctx);  }
    if (ctx2) { clReleaseContext(ctx2); }
    if (ctx3) { clReleaseContext(ctx3); }
    if (ctx4) { clReleaseContext(ctx4); }
    return 0;
}
