// test_reset.c
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
    
    printf("Resetting device...\n");
    if (ioctl(fd, GPGPU_IOCTL_RESET, NULL) < 0) {
        perror("RESET");
    } else {
        printf("Reset OK\n");
    }
    
    __u32 status;
    ioctl(fd, GPGPU_IOCTL_GET_STATUS, &status);
    printf("Status after reset: 0x%08x\n", status);
    
    close(fd);
    return 0;
}