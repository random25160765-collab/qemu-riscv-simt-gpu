/*
 * test_runner.c — VPU Verifier + Benchmark
 * 每个测试前清 VRAM，保证幂等
 * Copyright (c) 2024-2025, GPL v2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "state.h"
#include "gpgpu_core.h"
#include "scheduler.h"
#include "test_runner.h"

#define KERN_ADDR 0x500000

/* ============================================================
 * helpers
 * ============================================================ */
static uint8_t *read_file(const char *path, size_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st; fstat(fd, &st); *out_size = st.st_size;
    uint8_t *buf = malloc(st.st_size);
    if (!buf) { close(fd); return NULL; }
    if (read(fd, buf, st.st_size) != (ssize_t)st.st_size) { perror("read"); free(buf); close(fd); return NULL; }
    close(fd); return buf;
}
static size_t load_kernel(const char *path, GPGPUState *s, uint32_t addr) {
    size_t sz; uint8_t *buf = read_file(path, &sz);
    if (!buf) return 0;
    memcpy(s->vram_ptr + addr, buf, sz); free(buf); return sz;
}

/* ============================================================
 * verify: 功能验证, 无性能输出. 测试函数自己负责准备 VRAM 数据
 * ============================================================ */
static int verify(GPGPUState *s, const char *kern, uint32_t gx, uint32_t gy, uint32_t gz,
                   uint32_t bx, uint32_t by, uint32_t bz) {
    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0]=gx; s->kernel.grid_dim[1]=gy; s->kernel.grid_dim[2]=gz;
    s->kernel.block_dim[0]=bx; s->kernel.block_dim[1]=by; s->kernel.block_dim[2]=bz;
    load_kernel(kern, s, KERN_ADDR);
    return scheduler_run_kernel(s);
}

/* ============================================================
 * bench: 性能测试, 显示 MFLOPS. flops 预计算
 * ============================================================ */
static int bench(GPGPUState *s, uint64_t flops, const char *kern,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t bx, uint32_t by, uint32_t bz) {
    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0]=gx; s->kernel.grid_dim[1]=gy; s->kernel.grid_dim[2]=gz;
    s->kernel.block_dim[0]=bx; s->kernel.block_dim[1]=by; s->kernel.block_dim[2]=bz;
    load_kernel(kern, s, KERN_ADDR);
    s->fp_count = flops;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ret = scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    if (flops >= 1e9)
        printf("  %5.0fms  %6.1f GFLOPS  %lu flops  %u blocks\n", t*1e3, flops/t/1e9, (unsigned long)flops, gx*gy*gz);
    else
        printf("  %5.0fus  %6.1f MFLOPS  %lu flops  %u blocks\n", t*1e6, flops/t/1e6, (unsigned long)flops, gx*gy*gz);
    return ret;
}

/* macro for each test: reset VRAM, run, report */
#define TEST(name, fn) do { \
    printf("--- %s ---\n", name); \
    memset(s->vram_ptr, 0, 5*1024*1024); \
    int _e = fn(s); \
    if (_e) total_errors++; \
    printf("  %s\n", _e?"FAIL":"PASS"); \
} while(0)

#define BENCH(name, fn) do { \
    printf("--- %s ---\n", name); \
    memset(s->vram_ptr, 0, 5*1024*1024); \
    fn(s); \
} while(0)

/* ============================================================
 * Functional Tests
 * ============================================================ */
static int t_vector_add(GPGPUState *s) {
    uint32_t N=2048;
    for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i+1);((float*)(s->vram_ptr+0x200000))[i]=2.0f;}
    if(verify(s,"kernels/vecmul.bin",1,1,1,N,1,1))return 1;
    for(uint32_t i=0;i<N;i++) if(fabsf(((float*)(s->vram_ptr+0x300000))[i]-(float)(i+1)*2.0f)>1e-5f)return 1;
    return 0;
}
static int t_saxpy(GPGPUState *s) {
    uint32_t N=16384;float A=2.5f;
    *(uint32_t*)(s->vram_ptr)=N;*(float*)(s->vram_ptr+4)=A;
    srand48(42);
    float *yc=malloc(N*sizeof(float));
    for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(drand48()*100-50);yc[i]=((float*)(s->vram_ptr+0x200000))[i]=(float)(drand48()*100-50);}
    if(verify(s,"kernels/saxpy.bin",1,1,1,1,1,1)){free(yc);return 1;}
    int e=0;
    for(uint32_t i=0;i<N&&e<10;i++){float exp=A*((float*)(s->vram_ptr+0x100000))[i]+yc[i];if(fabsf(((float*)(s->vram_ptr+0x200000))[i]-exp)>1e-4f*fabsf(exp))e++;}
    free(yc);return e?1:0;
}
static int t_rv32m(GPGPUState *s) {
    uint32_t cs[][2]={{10,3},{10,(uint32_t)-3},{(uint32_t)-5,(uint32_t)-2},{100,0},{0x80000000,2},{0xFFFFFFFF,0xFFFFFFFF}};uint32_t N=6;
    *(uint32_t*)(s->vram_ptr)=N;memcpy(s->vram_ptr+0x100000,cs,sizeof(cs));
    if(verify(s,"kernels/rv32m.bin",1,1,1,1,1,1))return 1;
    for(uint32_t i=0;i<N;i++){int32_t a=cs[i][0],b=cs[i][1];uint32_t*o=(uint32_t*)(s->vram_ptr+0x300000)+i*8;
        uint32_t ex[]={(uint32_t)(a*b),(uint32_t)(((int64_t)a*(int64_t)b)>>32),(uint32_t)(((int64_t)a*(uint64_t)(uint32_t)b)>>32),(uint32_t)(((uint64_t)(uint32_t)a*(uint64_t)(uint32_t)b)>>32),b?(uint32_t)(a/b):(uint32_t)-1,b?((uint32_t)a/(uint32_t)b):0xFFFFFFFF,b?(uint32_t)(a%b):(uint32_t)a,b?((uint32_t)a%(uint32_t)b):(uint32_t)a};
        for(int j=0;j<8;j++)if(o[j]!=ex[j])return 1;}
    return 0;
}
static int t_rv32f(GPGPUState *s) {
    float fa=3.0f,fb=2.0f,fc=4.0f;int32_t i32=-5;uint32_t u32=7;
    memcpy(s->vram_ptr+0x100000,&fa,4);memcpy(s->vram_ptr+0x100004,&fb,4);memcpy(s->vram_ptr+0x100008,&fc,4);
    memcpy(s->vram_ptr+0x10000C,&i32,4);memcpy(s->vram_ptr+0x100010,&u32,4);
    if(verify(s,"kernels/rv32f.bin",1,1,1,1,1,1))return 1;
    float*fo=(float*)(s->vram_ptr+0x300000);uint32_t*io=(uint32_t*)(s->vram_ptr+0x300000);int e=0;
    #define CF(i,v) if(fabsf(fo[i]-(v))>1e-4f*fabsf(v)&&fabsf(fo[i]-(v))>1e-5f)e++
    #define CI(i,v) if(io[i]!=(uint32_t)(v))e++
    CF(0,fa+fb);CF(1,fa-fb);CF(2,fa*fb);CF(3,fa/fb);CF(4,sqrtf(fa));
    CF(5,fa*fb+fc);CF(6,fa*fb-fc);CF(7,-(fa*fb-fc));CF(8,-(fa*fb+fc));
    CF(9,3.0f);CF(10,-3.0f);CF(11,3.0f);CF(12,2.0f);CF(13,3.0f);
    CI(14,0);CI(15,0);CI(16,1);CI(17,1);CI(18,3);CI(19,3);
    CF(20,-5.0f);CF(21,7.0f);CI(22,7);CI(23,0x40400000);CI(24,0x40);CI(25,0x02);
    #undef CF #undef CI
    return e?1:0;
}
static int t_mem_access(GPGPUState *s) {
    uint8_t in[8]={0x7F,0x80,0xFF,0x00,0x34,0x12,0x78,0x56};memcpy(s->vram_ptr+0x100000,in,8);
    if(verify(s,"kernels/mem_access.bin",1,1,1,1,1,1))return 1;
    uint32_t*o=(uint32_t*)(s->vram_ptr+0x300000);
    #define C(i,v) if(o[i]!=(uint32_t)(v))return 1
    C(0,127);C(1,(int32_t)-128);C(2,(int32_t)-1);C(3,128);C(4,255);
    C(5,0x1234);C(6,0x5678);C(7,0x1234);C(8,0x00FF807F);
    C(9,0x42);C(10,0x4321);C(11,0x00FF807F);
    #undef C
    return 0;
}
static int t_matmul(GPGPUState *s) {
    uint32_t M=1,K=2,N=1;*(uint32_t*)(s->vram_ptr)=K;
    ((float*)(s->vram_ptr+0x100000))[0]=1;((float*)(s->vram_ptr+0x100000))[1]=2;
    ((float*)(s->vram_ptr+0x200000))[0]=3;((float*)(s->vram_ptr+0x200000))[1]=4;
    if(verify(s,"kernels/matmul.bin",M,1,1,N,1,1))return 1;
    float exp=1*3+2*4;return fabsf(((float*)(s->vram_ptr+0x300000))[0]-exp)>1e-3f?1:0;
}
static int t_dot_product(GPGPUState *s) {
    uint32_t N=32768;*(uint32_t*)(s->vram_ptr)=N;float sum=0;
    for(uint32_t i=0;i<N;i++){float a=(float)(i%100)*0.01f,b=(float)((i+1)%100)*0.01f;
        ((float*)(s->vram_ptr+0x100000))[i]=a;((float*)(s->vram_ptr+0x200000))[i]=b;sum+=a*b;}
    if(verify(s,"kernels/dot_product.bin",1,1,1,1,1,1))return 1;
    return fabsf(*(float*)(s->vram_ptr+0x4)-sum)>0.01f?1:0;
}
static int t_memcpy(GPGPUState *s) {
    uint32_t N=131072;*(uint32_t*)(s->vram_ptr)=N;
    for(uint32_t i=0;i<N;i++){((uint32_t*)(s->vram_ptr+0x100000))[i]=i;((uint32_t*)(s->vram_ptr+0x200000))[i]=0;}
    if(verify(s,"kernels/memcpy.bin",1,1,1,1,1,1))return 1;
    for(uint32_t i=0;i<N;i++)if(((uint32_t*)(s->vram_ptr+0x200000))[i]!=i)return 1;
    return 0;
}
static int t_vecmul(GPGPUState *s) {
    ((float*)(s->vram_ptr+0x100000))[0]=2.5f;((float*)(s->vram_ptr+0x200000))[0]=3.0f;
    if(verify(s,"kernels/vecmul.bin",1,1,1,1,1,1))return 1;
    return fabsf(((float*)(s->vram_ptr+0x300000))[0]-7.5f)>1e-5f?1:0;
}
static int t_scal_mul(GPGPUState *s) {
    ((float*)(s->vram_ptr+0x100000))[0]=4.0f;*(float*)(s->vram_ptr+0x400000)=0.75f;
    if(verify(s,"kernels/scal_mul.bin",1,1,1,1,1,1))return 1;
    return fabsf(((float*)(s->vram_ptr+0x200000))[0]-3.0f)>1e-5f?1:0;
}
static int t_gelu(GPGPUState *s) {
    float x=0.5f;((float*)(s->vram_ptr+0x100000))[0]=x;
    if(verify(s,"kernels/gelu.bin",1,1,1,1,1,1))return 1;
    float exp=x*(1.0f/(1.0f+expf(-1.702f*x)));
    return fabsf(((float*)(s->vram_ptr+0x200000))[0]-exp)>1e-3f?1:0;
}
static int t_softmax(GPGPUState *s) {
    uint32_t N=64;
    for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=((float)i-32.0f)*0.1f;
    if(verify(s,"kernels/softmax.bin",1,1,1,N,1,1))return 1;
    float sum=0;for(uint32_t i=0;i<N;i++)sum+=((float*)(s->vram_ptr+0x200000))[i];
    return fabsf(sum-1.0f)>0.02f?1:0;
}
static int t_softmax_norm(GPGPUState *s) {
    ((float*)(s->vram_ptr+0x200000))[0]=expf(1.0f);*(float*)(s->vram_ptr+0x400000)=expf(1.0f);
    if(verify(s,"kernels/softmax_norm.bin",1,1,1,1,1,1))return 1;
    return fabsf(((float*)(s->vram_ptr+0x300000))[0]-1.0f)>0.01f?1:0;
}
static int t_conv2d(GPGPUState *s) {
    uint32_t H=3,W=3,K=2;float in[9]={1,2,3,4,5,6,7,8,9},kr[4]={1,0,0,1};
    for(uint32_t i=0;i<H;i++)for(uint32_t j=0;j<W;j++)((float*)(s->vram_ptr+0x100000))[i*W+j]=in[i*W+j];
    for(uint32_t i=0;i<K;i++)for(uint32_t j=0;j<K;j++)((float*)(s->vram_ptr+0x200000))[i*K+j]=kr[i*K+j];
    if(verify(s,"kernels/conv2d.bin",H,W,1,K,1,1))return 1;
    float exp=6.0f;return fabsf(((float*)(s->vram_ptr+0x300000))[0]-exp)>1e-4f?1:0;
}

/* ============================================================
 * Parallel Benchmarks
 * ============================================================ */
static void b_vecmul_sizes(GPGPUState *s) {
    uint32_t sz[]={65536,262144,1048576};
    for(int si=0;si<3;si++){uint32_t N=sz[si];
        for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i%256);((float*)(s->vram_ptr+0x200000))[i]=3.0f;}
        printf("--- vecmul %u ---\n",N);bench(s,N,"kernels/vecmul.bin",N/32,1,1,32,1,1);}
}
static void b_matmul_sizes(GPGPUState *s) {
    int sz[]={128,256,512,1024};
    for(int si=0;si<4;si++){int M=sz[si],K=sz[si],N=sz[si];*(uint32_t*)(s->vram_ptr)=K;
        for(int i=0;i<M*K;i++)((float*)(s->vram_ptr+0x100000))[i]=1.0f;
        for(int i=0;i<K*N;i++)((float*)(s->vram_ptr+0x200000))[i]=1.0f;
        uint64_t fl=(uint64_t)M*N*K*2;printf("--- matmul %d ---\n",M);bench(s,fl,"kernels/matmul.bin",M,1,1,N,1,1);}
}
static void b_scal_mul(GPGPUState *s) {
    uint32_t N=65536;for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=(float)(i%256);*(float*)(s->vram_ptr+0x400000)=0.75f;
    printf("--- scal_mul %u ---\n",N);bench(s,N,"kernels/scal_mul.bin",N/32,1,1,32,1,1);
}
static void b_gelu(GPGPUState *s) {
    uint32_t N=65536;for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=(float)i*0.01f;
    printf("--- gelu %u ---\n",N);bench(s,N*6ULL,"kernels/gelu.bin",N/32,1,1,32,1,1);
}
static void b_softmax(GPGPUState *s) {
    uint32_t N=256;for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=((float)i-128.0f)*0.1f;
    printf("--- softmax %u ---\n",N);bench(s,(uint64_t)N*N*5,"kernels/softmax.bin",N,1,1,1,1,1);
}

/* ============================================================
 * entry
 * ============================================================ */
int test_runner_run(GPGPUState *s) {
    int total_errors = 0;

    printf("=== Functional Regression ===\n\n");
    TEST("Vector Add (multi-lane)", t_vector_add);
    TEST("SAXPY (serial)",         t_saxpy);
    TEST("RV32M Multiply/Divide",  t_rv32m);
    TEST("RV32F Full Coverage",    t_rv32f);
    TEST("Memory Access (lb/sb)",  t_mem_access);
    TEST("Matmul",                 t_matmul);
    TEST("Dot Product",            t_dot_product);
    TEST("Memcpy",                 t_memcpy);
    TEST("Vecmul (single)",        t_vecmul);
    TEST("Scal_mul",               t_scal_mul);
    TEST("GELU",                   t_gelu);
    TEST("Softmax",                t_softmax);
    TEST("Softmax_norm",           t_softmax_norm);
    TEST("Conv2d",                 t_conv2d);

    printf("\n=== Parallel Performance ===\n\n");
    BENCH("vecmul sizes",  b_vecmul_sizes);
    BENCH("matmul sizes",  b_matmul_sizes);
    BENCH("scal_mul 65536",b_scal_mul);
    BENCH("gelu 65536",    b_gelu);
    BENCH("softmax 256",   b_softmax);

    printf("\n===========================================\n");
    printf("Total errors: %d\n", total_errors);
    printf("===========================================\n");
    return total_errors;
}
