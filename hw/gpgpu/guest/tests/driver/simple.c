#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stdlib.h> 
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include "gpgpu_ioctl.h"

#define VRAM_SIZE       (64 * 1024 * 1024)  // 64MB

int main() {
    int fd = open("/dev/gpgpu0", O_RDWR);
    void *vram = mmap(NULL, 64*1024*1024, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    
    // 复位设备
    ioctl(fd, GPGPU_IOCTL_RESET, NULL);
    usleep(10000);
    
    // 读取并写入 kernel
    FILE *fp = fopen("bin/kernel/vector_add.bin", "rb");
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *kernel = malloc(size);
    fread(kernel, 1, size, fp);
    fclose(fp);
    memcpy(vram, kernel, size);
    
    // 准备测试数据
    float a = 3.14f;
    float b = 2.86f;
    memcpy(vram + 0x100000, &a, sizeof(a));
    memcpy(vram + 0x200000, &b, sizeof(b));
    printf("A[0] = %.2f, B[0] = %.2f, Expected C[0] = %.2f\n", a, b, a + b);
    
    // 启动 kernel
    struct gpgpu_kernel_params params = {
        .grid_dim = {1, 1, 1},
        .block_dim = {1, 1, 1},
        .kernel_addr = 0,
    };
    
    ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params);
    sleep(1);
    
    // 读取结果
    float c;
    memcpy(&c, vram + 0x300000, sizeof(c));
    printf("C[0] = %.2f\n", c);
    
    munmap(vram, 64*1024*1024);
    close(fd);
    return 0;
}