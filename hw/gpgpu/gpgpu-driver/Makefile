# GPGPU 驱动 Makefile

obj-m += gpgpu.o

KERNEL_DIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# 测试程序
TEST_PROGS := test_gpgpu
TEST_SRCS := test_gpgpu.c

all: modules $(TEST_PROGS)

modules:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

$(TEST_PROGS): %: %.c gpgpu_ioctl.h
	gcc -o $@ $< -Wall -O2

kernel: vector_add.bin

vector_add.bin: vector_add.S kernel.ld
	riscv64-linux-gnu-gcc -c -march=rv64g -O2 -nostdlib -ffreestanding vector_add.S -o vector_add.o
	riscv64-linux-gnu-ld -T kernel.ld vector_add.o -o vector_add.elf
	riscv64-linux-gnu-objcopy -O binary vector_add.elf vector_add.bin
	@echo "Kernel size: $$(stat -c%s vector_add.bin) bytes"

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f $(TEST_PROGS) *.o *.elf *.bin

install:
	sudo insmod gpgpu.ko

remove:
	sudo rmmod gpgpu

test: $(TEST_PROGS) vector_add.bin
	sudo ./test_gpgpu

dmesg:
	sudo dmesg -w | grep gpgpu

dmesg-clear:
	sudo dmesg -c > /dev/null

status:
	@ls -la /dev/gpgpu* 2>/dev/null || echo "No gpgpu devices found"
	@lsmod | grep gpgpu || echo "gpgpu module not loaded"

.PHONY: all modules clean install remove test dmesg dmesg-clear status kernel