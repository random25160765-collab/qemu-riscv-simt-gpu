# Makefile — RISC-V SIMT 解释器独立性能测试
#
# 目标:
#   make            — 编译解释器和所有 kernel，运行测试
#   make perf_test  — 编译解释器、kernel 并运行
#   make clean      — 清理构建产物

.PHONY: all clean perf_test kernels

# ============================================================
# 工具链
# ============================================================
CC       = gcc
RV_CC    = riscv64-unknown-elf-gcc
RV_OBJCOPY = riscv64-unknown-elf-objcopy

# ============================================================
# 编译选项
# ============================================================
CFLAGS   = -Wall -Wextra -O3 -mavx2 -g -fno-strict-aliasing
CFLAGS  += -I. -Icore -Iproto
LDFLAGS  = -lm -lrt -lpthread

# RV kernel 编译
RV_CFLAGS  = -march=rv32imf -mabi=ilp32 -nostdlib -Ttext=0x0
RV_CFLAGS += -Wl,--build-id=none

# ============================================================
# 源文件
# ============================================================
SRCS  = main.c
SRCS += test_runner.c
SRCS += core/engine.c
SRCS += core/scheduler.c
SRCS += core/fusion.c
SRCS += core/decode_trie.c
SRCS += core/lpfp.c
SRCS += core/memory.c
SRCS += core/proto.c

OBJS = $(SRCS:.c=.o)

# ============================================================
# Kernel 源文件和目标
# ============================================================
KERN_SRCS = $(wildcard kernels/*.S)
KERN_BINS = $(KERN_SRCS:.S=.bin)

# ============================================================
# 主目标
# ============================================================
all: perf_test

perf_test: $(KERN_BINS) perf_test.elf
	@echo ""
	@echo "=== Running Performance Tests ==="
	@echo ""
	./perf_test.elf

test: $(KERN_BINS) perf_test.elf
	./perf_test.elf

perf_test.elf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  LINK $@"

# ============================================================
# C 编译规则
# ============================================================
%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<
	@echo "  CC   $@"

# ============================================================
# Kernel 编译规则 (RISC-V asm → .bin)
# ============================================================
kernels/%.bin: kernels/%.S
	$(RV_CC) $(RV_CFLAGS) -o $@.elf $<
	$(RV_OBJCOPY) -O binary -j .text $@.elf $@
	rm -f $@.elf
	@echo "  RV_ASM $@"

# ============================================================
# 清理
# ============================================================
clean:
	rm -f perf_test.elf
	rm -f $(OBJS)
	rm -f $(KERN_BINS)
	@echo "  CLEAN done"

# ============================================================
# 仅编译 kernel（不跑测试）
# ============================================================
kernels: $(KERN_BINS)

# ============================================================
# 帮助
# ============================================================
help:
	@echo "Targets:"
	@echo "  make            — 编译并运行性能测试"
	@echo "  make kernels    — 仅编译 RISC-V test kernel (.S → .bin)"
	@echo "  make perf_test.elf — 仅编译 host 端解释器"
	@echo "  make clean      — 清理所有产物"
	@echo "  ./perf_test.elf — 直接运行已编译的测试"
