# VPU Simulator Test Suite
.PHONY: all clean test kernels help

CC       = gcc
RV_CC    = riscv64-unknown-elf-gcc
RV_OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS   = -Wall -Wextra -O3 -mavx2 -g -fno-strict-aliasing -I. -Icore -Iproto -I/usr/include/lua5.4
LDFLAGS  = -lm -lrt -lpthread -llua5.4 -ldl
RV_CFLAGS = -march=rv32imf -mabi=ilp32 -nostdlib -Ttext=0x0 -Wl,--build-id=none

SRCS  = main.c test_runner.c config.c
SRCS += core/engine.c core/scheduler.c core/fusion.c
SRCS += core/decode_trie.c core/lpfp.c core/memory.c core/proto.c
OBJS  = $(SRCS:.c=.o)

KERN_SRCS = $(wildcard kernels/*.S)
KERN_BINS = $(KERN_SRCS:.S=.bin)

all: test

test: $(KERN_BINS) perf_test.elf
	./perf_test.elf

perf_test.elf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

kernels/%.bin: kernels/%.S
	$(RV_CC) $(RV_CFLAGS) -o $@.elf $<
	$(RV_OBJCOPY) -O binary -j .text $@.elf $@
	rm -f $@.elf

kernels: $(KERN_BINS)

clean:
	rm -f perf_test.elf $(OBJS) $(KERN_BINS)

help:
	@echo "make        — build & run all tests"
	@echo "make test   — same"
	@echo "make clean  — remove build artifacts"
