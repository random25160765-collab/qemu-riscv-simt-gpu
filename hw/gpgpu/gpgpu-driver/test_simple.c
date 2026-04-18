#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "gpgpu_ioctl.h"

int main() {
    int fd = open("/dev/gpgpu0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    // 打印 ioctl 命令号
    printf("GPGPU_IOCTL_LAUNCH_PARAMS = 0x%lx\n", 
           (unsigned long)GPGPU_IOCTL_LAUNCH_PARAMS);
    printf("GPGPU_IOCTL_SET_GRID_DIM  = 0x%lx\n", 
           (unsigned long)GPGPU_IOCTL_SET_GRID_DIM);
    printf("GPGPU_IOCTL_GET_STATUS     = 0x%lx\n", 
           (unsigned long)GPGPU_IOCTL_GET_STATUS);
    
    // 测试 LAUNCH_PARAMS
    struct gpgpu_kernel_params params = {
        .grid_dim = {1, 1, 1},
        .block_dim = {1, 1, 1},
        .kernel_addr = 0,
        .args_addr = 0,
        .shared_mem = 0
    };
    
    printf("\nTesting LAUNCH_PARAMS...\n");
    if (ioctl(fd, GPGPU_IOCTL_LAUNCH_PARAMS, &params) < 0) {
        perror("LAUNCH_PARAMS");
    } else {
        printf("LAUNCH_PARAMS: OK\n");
    }
    
    close(fd);
    return 0;
}