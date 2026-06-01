/*
 * test_runner.c — VPU Test Framework (registration-based)
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
#include "scheduler.h"
#include "test_runner.h"

#define KERN_ADDR 0x500000
#define MAX_TESTS 64

static TestCase tests[MAX_TESTS];
static int n_tests = 0;
void test_register(TestCase t) { if (n_tests < MAX_TESTS) tests[n_tests++] = t; }

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
static void load_kernel(const char *path, GPGPUState *s, uint32_t addr) {
    size_t sz; uint8_t *buf = read_file(path, &sz);
    if (buf) { memcpy(s->vram_ptr + addr, buf, sz); free(buf); s->kern_size = (uint32_t)sz; }
}

static uint64_t g_bp[3];
static void do_register(void);
static int run_one(GPGPUState *s, TestCase *t) {
    memset(s->vram_ptr, 0, 5*1024*1024);
    memcpy(g_bp, t->params, sizeof(g_bp));
    if (t->setup) t->setup(s);
    s->kernel.kernel_addr = KERN_ADDR;
    memcpy(s->kernel.grid_dim, t->grid, sizeof(t->grid));
    memcpy(s->kernel.block_dim, t->block, sizeof(t->block));
    if (t->kernel) load_kernel(t->kernel, s, KERN_ADDR);
    s->fp_count = t->flops;
    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    int ret = scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    if (ret) return 1;
    double us = (T1.tv_sec-T0.tv_sec)*1e6 + (T1.tv_nsec-T0.tv_nsec)*1e-3;
    int err = t->check ? t->check(s) : 0;
    if (t->bench && t->flops) {
        double mf = us > 0 ? t->flops / us : 0;
        printf("  %5.0fus  %6.1f MFLOPS", us, mf);
    }
    printf("  %s\n", err ? "FAIL" : "PASS");
    return err ? 1 : 0;
}

int test_run(GPGPUState *s, const char *group, const char *filter) {
    do_register();
    int errors = 0;
    printf("=== %s ===\n\n", group ? group : "All");
    for (int i = 0; i < n_tests; i++) {
        TestCase *t = &tests[i];
        if (filter && !strstr(t->name, filter)) continue;
        if (group && !strcmp(group, "func") && t->bench) continue;
        if (group && !strcmp(group, "bench") && !t->bench) continue;
        printf("--- %s ---\n", t->name);
        errors += run_one(s, t);
    }
    printf("\n");
    return errors;
}

/* ============================================================
 * Functional tests
 * ============================================================ */
static void s_vecmul(GPGPUState *s) { for(uint32_t i=0;i<2048;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i+1);((float*)(s->vram_ptr+0x200000))[i]=2.0f;} }
static int c_vecmul(GPGPUState *s) { for(uint32_t i=0;i<2048;i++) if(fabsf(((float*)(s->vram_ptr+0x300000))[i]-(float)(i+1)*2.0f)>1e-5f)return 1; return 0; }
static void s_saxpy(GPGPUState *s) { uint32_t N=16384;srand48(42);*(uint32_t*)s->vram_ptr=N;*(float*)(s->vram_ptr+4)=2.5f;for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(drand48()*100-50);((float*)(s->vram_ptr+0x200000))[i]=(float)(drand48()*100-50);} }
static int c_saxpy(GPGPUState *s) { uint32_t N=16384;float A=2.5f;srand48(42);float*y=malloc(N*4);for(uint32_t i=0;i<N;i++){drand48(); /*skip X[i]*/ y[i]=(float)(drand48()*100-50);} int e=0;for(uint32_t i=0;i<N&&e<10;i++){float exp=A*((float*)(s->vram_ptr+0x100000))[i]+y[i];if(fabsf(((float*)(s->vram_ptr+0x200000))[i]-exp)>1e-4f*fabsf(exp))e++;}free(y);return e?1:0;}
static void s_rv32m(GPGPUState *s) { uint32_t cs[][2]={{10,3},{10,(uint32_t)-3},{(uint32_t)-5,(uint32_t)-2},{100,0},{0x80000000,2},{0xFFFFFFFF,0xFFFFFFFF}};*(uint32_t*)s->vram_ptr=6;memcpy(s->vram_ptr+0x100000,cs,sizeof(cs));}
static int c_rv32m(GPGPUState *s) { uint32_t cs[][2]={{10,3},{10,(uint32_t)-3},{(uint32_t)-5,(uint32_t)-2},{100,0},{0x80000000,2},{0xFFFFFFFF,0xFFFFFFFF}};for(uint32_t i=0;i<6;i++){int32_t a=cs[i][0],b=cs[i][1];uint32_t*o=(uint32_t*)(s->vram_ptr+0x300000)+i*8;uint32_t ex[]={(uint32_t)(a*b),(uint32_t)(((int64_t)a*(int64_t)b)>>32),(uint32_t)(((int64_t)a*(uint64_t)(uint32_t)b)>>32),(uint32_t)(((uint64_t)(uint32_t)a*(uint64_t)(uint32_t)b)>>32),b?(uint32_t)(a/b):(uint32_t)-1,b?((uint32_t)a/(uint32_t)b):0xFFFFFFFF,b?(uint32_t)(a%b):(uint32_t)a,b?((uint32_t)a%(uint32_t)b):(uint32_t)a};for(int j=0;j<8;j++)if(o[j]!=ex[j])return 1;}return 0;}
static void s_rv32f(GPGPUState *s) { float fa=3,fb=2,fc=4;int32_t i32=-5;uint32_t u32=7;memcpy(s->vram_ptr+0x100000,&fa,4);memcpy(s->vram_ptr+0x100004,&fb,4);memcpy(s->vram_ptr+0x100008,&fc,4);memcpy(s->vram_ptr+0x10000C,&i32,4);memcpy(s->vram_ptr+0x100010,&u32,4); }
static int c_rv32f(GPGPUState *s) { float*fo=(float*)(s->vram_ptr+0x300000);uint32_t*io=(uint32_t*)(s->vram_ptr+0x300000);int e=0;float fa=3,fb=2,fc=4;
#define CF(i,v) if(fabsf(fo[i]-(v))>1e-4f*fabsf(v)&&fabsf(fo[i]-(v))>1e-5f)e++
#define CI(i,v) if(io[i]!=(uint32_t)(v))e++
CF(0,fa+fb);CF(1,fa-fb);CF(2,fa*fb);CF(3,fa/fb);CF(4,sqrtf(fa));CF(5,fa*fb+fc);CF(6,fa*fb-fc);CF(7,-(fa*fb-fc));CF(8,-(fa*fb+fc));CF(9,3);CF(10,-3);CF(11,3);CF(12,2);CF(13,3);CI(14,0);CI(15,0);CI(16,1);CI(17,1);CI(18,3);CI(19,3);CF(20,-5);CF(21,7);CI(22,7);CI(23,0x40400000);CI(24,0x40);CI(25,0x02);
#undef CF #undef CI
return e?1:0;}
static void s_mem_acc(GPGPUState *s) { uint8_t in[8]={0x7F,0x80,0xFF,0x00,0x34,0x12,0x78,0x56};memcpy(s->vram_ptr+0x100000,in,8); }
static int c_mem_acc(GPGPUState *s) { uint32_t*o=(uint32_t*)(s->vram_ptr+0x300000);return(o[0]!=127||o[1]!=(uint32_t)(int32_t)-128||o[2]!=(uint32_t)-1||o[3]!=128||o[4]!=255||o[5]!=0x1234||o[6]!=0x5678||o[7]!=0x1234||o[8]!=0x00FF807F||o[9]!=0x42||o[10]!=0x4321||o[11]!=0x00FF807F);}
static void s_matmul_f(GPGPUState *s) { *(uint32_t*)s->vram_ptr=2;((float*)(s->vram_ptr+0x100000))[0]=1;((float*)(s->vram_ptr+0x100000))[1]=2;((float*)(s->vram_ptr+0x200000))[0]=3;((float*)(s->vram_ptr+0x200000))[1]=4; }
static int c_matmul_f(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x300000))[0]-11)>1e-3f; }
static void s_dot(GPGPUState *s) { uint32_t N=32768;*(uint32_t*)s->vram_ptr=N;for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i%100)*0.01f;((float*)(s->vram_ptr+0x200000))[i]=(float)((i+1)%100)*0.01f;} }
static int c_dot(GPGPUState *s) { float sum=0;for(uint32_t i=0;i<32768;i++)sum+=(float)(i%100)*0.01f*(float)((i+1)%100)*0.01f;return fabsf(*(float*)(s->vram_ptr+4)-sum)>0.01f;}
static void s_memcpy(GPGPUState *s) { uint32_t N=131072;*(uint32_t*)s->vram_ptr=N;for(uint32_t i=0;i<N;i++){((uint32_t*)(s->vram_ptr+0x100000))[i]=i;((uint32_t*)(s->vram_ptr+0x200000))[i]=0;} }
static int c_memcpy(GPGPUState *s) { for(uint32_t i=0;i<131072;i++)if(((uint32_t*)(s->vram_ptr+0x200000))[i]!=i)return 1;return 0; }
static void s_v1(GPGPUState *s) { ((float*)(s->vram_ptr+0x100000))[0]=2.5f;((float*)(s->vram_ptr+0x200000))[0]=3.0f; }
static int c_v1(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x300000))[0]-7.5f)>1e-5f; }
static void s_s1(GPGPUState *s) { ((float*)(s->vram_ptr+0x100000))[0]=4.0f;*(float*)(s->vram_ptr+0x400000)=0.75f; }
static int c_s1(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x200000))[0]-3.0f)>1e-5f; }
static void s_gelu(GPGPUState *s) { ((float*)(s->vram_ptr+0x100000))[0]=0.5f; }
static int c_gelu(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x200000))[0]-0.5f*(1/(1+expf(-1.702*0.5))))>1e-3f; }
static void s_softmax(GPGPUState *s) { for(uint32_t i=0;i<64;i++)((float*)(s->vram_ptr+0x100000))[i]=((float)i-32.0f)*0.1f; }
static int c_softmax(GPGPUState *s) { float sum=0;for(uint32_t i=0;i<64;i++)sum+=((float*)(s->vram_ptr+0x200000))[i];return fabsf(sum-1.0f)>0.02f; }
static void s_snorm(GPGPUState *s) { ((float*)(s->vram_ptr+0x200000))[0]=expf(1.0f);*(float*)(s->vram_ptr+0x400000)=expf(1.0f); }
static int c_snorm(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x300000))[0]-1.0f)>0.01f; }
static void s_conv(GPGPUState *s) { float in[]={1,2,3,4,5,6,7,8,9},kr[]={1,0,0,1};for(int i=0;i<9;i++)((float*)(s->vram_ptr+0x100000))[i]=in[i];for(int i=0;i<4;i++)((float*)(s->vram_ptr+0x200000))[i]=kr[i]; }
static int c_conv(GPGPUState *s) { return fabsf(((float*)(s->vram_ptr+0x300000))[0]-6.0f)>1e-4f; }

/* ============================================================
 * Bench setups (read params from g_bp[] set by run_one)
 * ============================================================ */
static void s_bv(GPGPUState *s) { uint32_t N=(uint32_t)g_bp[0]; for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i%256);((float*)(s->vram_ptr+0x200000))[i]=3.0f;} }
static void s_bm(GPGPUState *s) { uint32_t M=(uint32_t)g_bp[0],K=(uint32_t)g_bp[1],N=(uint32_t)g_bp[2]; *(uint32_t*)s->vram_ptr=K; for(uint32_t i=0;i<M*K;i++)((float*)(s->vram_ptr+0x100000))[i]=1.0f; for(uint32_t i=0;i<K*N;i++)((float*)(s->vram_ptr+0x200000))[i]=1.0f; }
static void s_bs(GPGPUState *s) { uint32_t N=(uint32_t)g_bp[0]; for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=(float)(i%256); *(float*)(s->vram_ptr+0x400000)=0.75f; }
static void s_bgelu(GPGPUState *s) { uint32_t N=(uint32_t)g_bp[0]; for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=(float)i*0.01f; }
static void s_bsoftmax(GPGPUState *s) { uint32_t N=(uint32_t)g_bp[0]; for(uint32_t i=0;i<N;i++)((float*)(s->vram_ptr+0x100000))[i]=((float)i-128.0f)*0.1f; }
static int c_ok(GPGPUState *s) { (void)s; return 0; }

#define B(name, kern, gx, gy, gz, bx, by, bz, fl, setup, p0, p1, p2) \
    test_register((TestCase){name, kern, {gx,gy,gz}, {bx,by,bz}, setup, c_ok, fl, {p0,p1,p2}, true, false})

/* ============================================================
 * Native C comparison
 * ============================================================ */
static void run_native_vecmul(GPGPUState *s) {
    uint32_t N=65536,nw=N/32;struct timespec T0,T1;
    for(uint32_t i=0;i<N;i++){((float*)(s->vram_ptr+0x100000))[i]=(float)(i%256);((float*)(s->vram_ptr+0x200000))[i]=3.0f;}
    s->kernel.kernel_addr=KERN_ADDR;s->kernel.grid_dim[0]=nw;s->kernel.grid_dim[1]=1;s->kernel.grid_dim[2]=1;s->kernel.block_dim[0]=1;s->kernel.block_dim[1]=1;s->kernel.block_dim[2]=1;
    load_kernel("kernels/vecmul.bin",s,KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC,&T0);scheduler_run_kernel(s);clock_gettime(CLOCK_MONOTONIC,&T1);
    double ti=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    volatile float sum=0;clock_gettime(CLOCK_MONOTONIC,&T0);
    for(uint32_t w=0;w<nw;w++){uint32_t g[32*32]={0};for(int l=0;l<32;l++)g[6*32+l]=w*32+l;for(int i=0;i<32;i++){uint32_t t=g[6*32+i];float a=*(float*)(s->vram_ptr+0x100000+t*4);float b=*(float*)(s->vram_ptr+0x200000+t*4);*(float*)(s->vram_ptr+0x300000+t*4)=a*b;sum+=a*b;}}
    clock_gettime(CLOCK_MONOTONIC,&T1);double tn=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    printf("  Interp: %5.0fus  Native: %5.0fus  Ratio: %.1fx\n",ti*1e6,tn*1e6,ti/tn);(void)sum;
}
static void run_native_matmul(GPGPUState *s) {
    int M=128,K=128,N=128;struct timespec T0,T1;
    *(uint32_t*)s->vram_ptr=K;for(int i=0;i<M*K;i++)((float*)(s->vram_ptr+0x100000))[i]=1.0f;for(int i=0;i<K*N;i++)((float*)(s->vram_ptr+0x200000))[i]=1.0f;
    s->kernel.kernel_addr=KERN_ADDR;s->kernel.grid_dim[0]=M;s->kernel.grid_dim[1]=1;s->kernel.grid_dim[2]=1;s->kernel.block_dim[0]=N;s->kernel.block_dim[1]=1;s->kernel.block_dim[2]=1;
    load_kernel("kernels/matmul.bin",s,KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC,&T0);scheduler_run_kernel(s);clock_gettime(CLOCK_MONOTONIC,&T1);
    double ti=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    volatile float sum=0;float*A=(float*)(s->vram_ptr+0x100000),*B=(float*)(s->vram_ptr+0x200000),*C=(float*)(s->vram_ptr+0x300000);memset(C,0,M*N*4);
    clock_gettime(CLOCK_MONOTONIC,&T0);
    for(int row=0;row<M;row++)for(int k=0;k<K;k++){float aik=A[row*K+k];float*cr=C+row*N,*br=B+k*N;for(int col=0;col<N;col++)cr[col]+=aik*br[col];}
    for(int i=0;i<M*N;i++)sum+=C[i];clock_gettime(CLOCK_MONOTONIC,&T1);
    double tn=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)*1e-9;
    printf("  Interp: %5.0fus  Native: %5.0fus  Ratio: %.1fx\n",ti*1e6,tn*1e6,ti/tn);(void)sum;
}
void test_run_native(GPGPUState *s) {
    printf("=== Native C Comparison ===\n\n");
    printf("--- vecmul ---\n");run_native_vecmul(s);
    printf("--- matmul ---\n");run_native_matmul(s);
    printf("\n");
}

/* ============================================================
 * register all
 * ============================================================ */
static int g_registered = 0;
static void do_register(void) {
    if (g_registered) return; g_registered = 1;

    /* functional */
    test_register((TestCase){"Vector Add","kernels/vecmul.bin",{1,1,1},{2048,1,1},s_vecmul,c_vecmul});
    test_register((TestCase){"SAXPY","kernels/saxpy.bin",{1,1,1},{1,1,1},s_saxpy,c_saxpy});
    test_register((TestCase){"RV32M","kernels/rv32m.bin",{1,1,1},{1,1,1},s_rv32m,c_rv32m});
    test_register((TestCase){"RV32F","kernels/rv32f.bin",{1,1,1},{1,1,1},s_rv32f,c_rv32f});
    test_register((TestCase){"Mem Access","kernels/mem_access.bin",{1,1,1},{1,1,1},s_mem_acc,c_mem_acc});
    test_register((TestCase){"Matmul","kernels/matmul.bin",{1,1,1},{1,1,1},s_matmul_f,c_matmul_f});
    test_register((TestCase){"Dot Product","kernels/dot_product.bin",{1,1,1},{1,1,1},s_dot,c_dot});
    test_register((TestCase){"Memcpy","kernels/memcpy.bin",{1,1,1},{1,1,1},s_memcpy,c_memcpy});
    test_register((TestCase){"Vecmul","kernels/vecmul.bin",{1,1,1},{1,1,1},s_v1,c_v1});
    test_register((TestCase){"Scal_mul","kernels/scal_mul.bin",{1,1,1},{1,1,1},s_s1,c_s1});
    test_register((TestCase){"GELU","kernels/gelu.bin",{1,1,1},{1,1,1},s_gelu,c_gelu});
    test_register((TestCase){"Softmax","kernels/softmax.bin",{1,1,1},{64,1,1},s_softmax,c_softmax});
    test_register((TestCase){"SoftmaxNorm","kernels/softmax_norm.bin",{1,1,1},{1,1,1},s_snorm,c_snorm});
    test_register((TestCase){"Conv2d","kernels/conv2d.bin",{3,3,1},{2,1,1},s_conv,c_conv});

    /* benchmarks */
    B("vecmul 64K", "kernels/vecmul.bin", 2048,1,1,32,1,1, 65536, s_bv, 65536,0,0);
    B("vecmul 256K","kernels/vecmul.bin", 8192,1,1,32,1,1, 262144, s_bv, 262144,0,0);
    B("vecmul 1M",  "kernels/vecmul.bin", 32768,1,1,32,1,1, 1048576, s_bv, 1048576,0,0);
    B("matmul 128", "kernels/matmul.bin",128,1,1,128,1,1,4194304ULL,s_bm,128,128,128);
    B("matmul 256", "kernels/matmul.bin",256,1,1,256,1,1,33554432ULL,s_bm,256,256,256);
    B("matmul 512", "kernels/matmul.bin",512,1,1,512,1,1,268435456ULL,s_bm,512,512,512);
    B("matmul 1024","kernels/matmul.bin",1024,1,1,1024,1,1,2147483648ULL,s_bm,1024,1024,1024);
    B("scal_mul 64K","kernels/scal_mul.bin",2048,1,1,32,1,1,65536,s_bs,65536,0,0);
    B("gelu 64K",   "kernels/gelu.bin",2048,1,1,32,1,1,393216,s_bgelu,65536,0,0);
    B("softmax 256","kernels/softmax.bin",256,1,1,1,1,1,327680,s_bsoftmax,256,0,0);
}

void test_run_all(GPGPUState *s) {
    do_register();
    int errors = 0;
    errors += test_run(s, "func", NULL);
    errors += test_run(s, "bench", NULL);
    test_run_native(s);
    printf("===========================================\nTotal errors: %d\n===========================================\n", errors);
}
