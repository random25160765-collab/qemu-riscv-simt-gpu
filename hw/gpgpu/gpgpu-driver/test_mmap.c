/*
 * GPGPU VRAM mmap 测试程序
 * 
 * 编译: gcc -o test_mmap test_mmap.c -Wall
 * 运行: sudo ./test_mmap
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#define DEVICE_PATH     "/dev/gpgpu0"
#define VRAM_SIZE       (64 * 1024 * 1024)  // 64MB
#define TEST_SIZE       4096                // 测试 4KB

int main(int argc, char *argv[])
{
    int fd;
    void *vram;
    int i;
    int ret = 0;

    printf("=== GPGPU VRAM mmap Test ===\n\n");

    // 1. 打开设备
    printf("Opening device: %s\n", DEVICE_PATH);
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("open failed");
        return 1;
    }
    printf("  -> fd = %d\n\n", fd);

    // 2. mmap VRAM
    printf("Mapping VRAM (size = %d bytes)...\n", VRAM_SIZE);
    vram = mmap(NULL, VRAM_SIZE, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return 1;
    }
    printf("  -> vram virtual address: %p\n\n", vram);

    // 3. 写入测试数据
    printf("Writing test pattern to VRAM...\n");
    char *test_data = (char *)vram;
    for (i = 0; i < TEST_SIZE; i++) {
        test_data[i] = (char)(i & 0xFF);
    }
    printf("  -> Wrote %d bytes\n\n", TEST_SIZE);

    // 4. 验证数据
    printf("Verifying data...\n");
    int errors = 0;
    for (i = 0; i < TEST_SIZE; i++) {
        char expected = (char)(i & 0xFF);
        if (test_data[i] != expected) {
            printf("  Mismatch at offset 0x%x: expected 0x%02x, got 0x%02x\n",
                   i, expected, test_data[i]);
            errors++;
            if (errors > 10) {
                printf("  ... too many errors\n");
                break;
            }
        }
    }
    
    if (errors == 0) {
        printf("  -> All %d bytes verified OK!\n\n", TEST_SIZE);
    } else {
        printf("  -> %d errors found\n\n", errors);
        ret = 1;
    }

    // 5. 显示前 64 字节
    printf("First 64 bytes of VRAM:\n");
    for (i = 0; i < 64; i++) {
        printf("%02x ", (unsigned char)test_data[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    printf("\n");

    // 6. 清零测试（可选）
    printf("Zeroing first 4KB of VRAM...\n");
    memset(vram, 0, TEST_SIZE);
    printf("  -> Done\n\n");

    // 7. 清理
    printf("Unmapping VRAM...\n");
    munmap(vram, VRAM_SIZE);
    
    printf("Closing device...\n");
    close(fd);

    printf("\n=== Test %s ===\n", ret == 0 ? "PASSED" : "FAILED");
    return ret;
}