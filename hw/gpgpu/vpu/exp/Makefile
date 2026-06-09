# VPU Simulator Test Suite
.PHONY: all clean test kernels help format format-check asan
.PHONY: libengine.a go-build go-test go-all

CMD    ?= func
FILTER ?=

format:
	@./scripts/format.sh

format-check:
	@./scripts/format.sh --check

CC       = gcc
RV_CC    = riscv64-unknown-elf-gcc
RV_OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS   = -O3 -mavx2 -g -fno-strict-aliasing -I. -Icore -Ilib -Iproto -I/usr/include/lua5.4
LDFLAGS  = -lm -lrt -lpthread -llua5.4 -ldl
RV_CFLAGS = -march=rv32imfv -mabi=ilp32 -nostdlib -Ttext=0x0 -Wl,--build-id=none

SRCS  = main.c test_runner.c config.c stats.c
SRCS += tests/func.c tests/bench.c tests/native.c tests/register.c
SRCS += core/engine.c core/scheduler.c core/vram_alloc.c
SRCS += core/decode_trie.c core/vram.c core/proto.c core/table.c lib/lpfp.c
# vpu/vpu.c tcu/tcu.c removed — dead code, replaced by inst/rvv/
OBJS  = $(SRCS:.c=.o)

# ── libengine.a: C engine for Go scheduler ──────────────────
# excludes scheduler.c, test_runner.c, tests/*.c, main.c
LIB_SRCS  = config.c stats.c
LIB_SRCS += core/engine.c core/engine_bridge.c core/vram_alloc.c
LIB_SRCS += core/decode_trie.c core/vram.c core/proto.c core/table.c lib/lpfp.c
# vpu/vpu.c tcu/tcu.c removed — dead code, replaced by inst/rvv/
LIB_OBJS  = $(LIB_SRCS:.c=.o)
LIB_CFLAGS = -Wall -Wextra -O2 -g -fno-strict-aliasing -I. -Icore -Ilib -Iproto -I/usr/include/lua5.4

libengine.a: $(LIB_OBJS)
	ar rcs $@ $^

$(LIB_OBJS): %.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

# ── Go scheduler ────────────────────────────────────────────
GOSCHED = gosched
GOFLAGS = CGO_ENABLED=1

go-build: libengine.a kernels
	cd $(GOSCHED) && $(GOFLAGS) go build -o ../gpu_sched .

go-test: libengine.a kernels
	cd $(GOSCHED) && $(GOFLAGS) go test -v ./...

go-all: go-build
	./gpu_sched

# ── Existing targets ────────────────────────────────────────

KERN_SRCS = $(wildcard kernels/*.S)
KERN_BINS = $(KERN_SRCS:.S=.bin)

# ── ISA handler 生成 (inst/ 子目录) ──────────────────────────
.PHONY: inst-gen inst-clean

inst-gen:
	@$(MAKE) -C inst

inst-clean:
	@$(MAKE) -C inst clean

# 编译前必须先生成 handler + dispatch 表
# engine.o 包含生成的 dispatch 头文件, 必须先于编译生成 (| = order-only)
$(OBJS): | inst-gen
$(LIB_OBJS): | inst-gen

all: test

test: $(KERN_BINS) perf_test.elf
	./perf_test.elf

perf_test.elf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 改 gpu_config.lua (perf 开关) 后自动重编 engine.c
core/engine.o: gpu_config.lua
# RVV handler 池中未分配 dispatch 条目的 label 产生 -Wunused-label,
# 加 decode 字段后自动消除。暂时局部关闭 warning。
core/engine.o: LIB_CFLAGS += -Wno-unused-label -Wno-unused-variable -Wno-type-limits

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

kernels/%.bin: kernels/%.S tools/kerncheck | inst-gen
	@echo "  ASM    $<"
	@rm -f $<.tmp.S
	@python3 tools/customasm.py $< $<.tmp.S
	@$(RV_CC) $(RV_CFLAGS) -o $@.elf $<.tmp.S
	@$(RV_OBJCOPY) -O binary -j .text $@.elf $@
	@rm -f $@.elf $<.tmp.S
	@./tools/kerncheck $@

tools/kerncheck: tools/kerncheck.c core/dispatch.h core/decode_trie.c core/inst.h core/utils.h | inst-gen
	@echo "  CC     tools/kerncheck"
	@$(CC) -Wall -O2 -I. -Icore -o $@ tools/kerncheck.c core/decode_trie.c -lm

kernels: $(KERN_BINS)

asan: CFLAGS = -Wall -Wextra -O0 -g -fsanitize=address -fno-omit-frame-pointer -I. -Icore -Iproto -I/usr/include/lua5.4
asan: LDFLAGS = -lm -lrt -lpthread -llua5.4 -ldl -fsanitize=address
asan: clean $(KERN_BINS) perf_test.elf
	@echo "=== ASAN $(CMD) ==="
	./perf_test.elf $(CMD) $(FILTER)

clean: inst-clean
	rm -f perf_test.elf $(OBJS) $(KERN_BINS) vpu_asan

count:
	@echo ".c:";  find . -name '*.c'  | xargs wc -l | tail -1
	@echo ".h:";  find . -name '*.h'  | xargs wc -l | tail -1
	@echo ".S:";  find . -name '*.S'  | xargs wc -l | tail -1
	@echo ".lua:"; find . -name '*.lua' | xargs wc -l | tail -1
	@echo ".py:";  find . -name '*.py'  | xargs wc -l | tail -1
	@echo ".sh:";  find . -name '*.sh'  | xargs wc -l | tail -1
	@echo ".go:";  find . -name '*.go'  | xargs wc -l | tail -1
	@echo ""; find . \( -name '*.c' -o -name '*.h' -o -name '*.lua' -o -name '*.py' -o -name '*.sh' -o -name '*.go' \) -exec wc -l {} + | sort -rn | head -15

help:
	@echo "make        — build & run all tests"
	@echo "make test   — same"
	@echo "make clean        — remove build artifacts"
	@echo "make count        — count lines of code"
	@echo "make format       — auto-format all C/H files"
	@echo "make format-check — check formatting (dry-run)"
