#!/bin/bash

set -e

echo "=== Building GPU Kernel ==="

# 编译汇编
riscv64-linux-gnu-gcc -c -march=rv64g -O2 -nostdlib -ffreestanding vector_add.S -o vector_add.o

# 链接
riscv64-linux-gnu-ld -T kernel.ld vector_add.o -o vector_add.elf

# 提取二进制
riscv64-linux-gnu-objcopy -O binary vector_add.elf vector_add.bin

# 显示大小
SIZE=$(stat -c%s vector_add.bin)
echo "Kernel size: $SIZE bytes"

# 显示十六进制（前 64 字节）
echo "First 64 bytes (hex):"
xxd -l 64 vector_add.bin

echo "=== Done ==="