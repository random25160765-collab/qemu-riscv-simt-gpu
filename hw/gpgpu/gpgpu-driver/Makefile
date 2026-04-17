# Makefile - GPGPU Linux Driver
obj-m += gpgpu.o

# 如果只有一个 .c 文件，这样就可以了
KERNEL_DIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean

# 额外的便利目标
install:
	sudo insmod gpgpu.ko

remove:
	sudo rmmod gpgpu
	
dmesg:
	sudo dmesg -w | grep gpgpu

.PHONY: all clean install remove info dmesg