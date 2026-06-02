# VPU Simulator Test Suite
.PHONY: all clean test kernels help format format-check asan

CMD    ?= func
FILTER ?=

format:
	@./scripts/format.sh

format-check:
	@./scripts/format.sh --check

CC       = gcc
RV_CC    = riscv64-unknown-elf-gcc
RV_OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS   = -Wall -Wextra -O3 -mavx2 -g -fno-strict-aliasing -I. -Icore -Ivpu -Itcu -Iproto -I/usr/include/lua5.4
LDFLAGS  = -lm -lrt -lpthread -llua5.4 -ldl
RV_CFLAGS = -march=rv32imf -mabi=ilp32 -nostdlib -Ttext=0x0 -Wl,--build-id=none

SRCS  = main.c test_runner.c config.c stats.c
SRCS += tests/func.c tests/bench.c tests/native.c tests/register.c
SRCS += core/engine.c core/scheduler.c core/vram_alloc.c
SRCS += core/decode_trie.c core/lpfp.c core/memory.c core/proto.c core/table.c
SRCS += vpu/vpu.c tcu/tcu.c
OBJS  = $(SRCS:.c=.o)

KERN_SRCS = $(wildcard kernels/*.S)
KERN_BINS = $(KERN_SRCS:.S=.bin)

all: test

test: $(KERN_BINS) perf_test.elf
	./perf_test.elf

perf_test.elf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 改 gpu_config.lua (perf 开关) 后自动重编 engine.c
core/engine.o: gpu_config.lua

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

kernels/%.bin: kernels/%.S tools/kerncheck
	@echo "  ASM    $<"
	@rm -f $<.tmp.S
	@python3 tools/customasm.py $< $<.tmp.S
	@$(RV_CC) $(RV_CFLAGS) -o $@.elf $<.tmp.S
	@$(RV_OBJCOPY) -O binary -j .text $@.elf $@
	@rm -f $@.elf $<.tmp.S
	@./tools/kerncheck $@

tools/kerncheck: tools/kerncheck.c core/dispatch.h core/decode_trie.c core/inst.h core/utils.h
	@echo "  CC     tools/kerncheck"
	@$(CC) -Wall -O2 -I. -Icore -o $@ tools/kerncheck.c core/decode_trie.c -lm

kernels: $(KERN_BINS)

asan: CFLAGS = -Wall -Wextra -O0 -g -fsanitize=address -fno-omit-frame-pointer -I. -Icore -Ivpu -Itcu -Iproto -I/usr/include/lua5.4
asan: LDFLAGS = -lm -lrt -lpthread -llua5.4 -ldl -fsanitize=address
asan: clean $(KERN_BINS) perf_test.elf
	@echo "=== ASAN $(CMD) ==="
	./perf_test.elf $(CMD) $(FILTER)

clean:
	rm -f perf_test.elf $(OBJS) $(KERN_BINS) vpu_asan

count:
	@echo ".c:";  find . -name '*.c'  | xargs wc -l | tail -1
	@echo ".h:";  find . -name '*.h'  | xargs wc -l | tail -1
	@echo ".S:";  find . -name '*.S'  | xargs wc -l | tail -1
	@echo ".lua:"; find . -name '*.lua' | xargs wc -l | tail -1
	@echo ".py:";  find . -name '*.py'  | xargs wc -l | tail -1
	@echo ".sh:";  find . -name '*.sh'  | xargs wc -l | tail -1
	@echo ""; find . \( -name '*.c' -o -name '*.h' -o -name '*.lua' -o -name '*.py' -o -name '*.sh' \) -exec wc -l {} + | sort -rn | head -15

help:
	@echo "make        — build & run all tests"
	@echo "make test   — same"
	@echo "make clean        — remove build artifacts"
	@echo "make count        — count lines of code"
	@echo "make format       — auto-format all C/H files"
	@echo "make format-check — check formatting (CI)"
