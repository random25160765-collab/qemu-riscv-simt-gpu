# conflux-runtime/Makefile
# ==============================
# 生成 libconflux.{a,so} 和测试可执行文件
#
# 用法（host 交叉编译给 RISC-V guest）：
#   make CROSS_COMPILE=riscv64-linux-gnu-
#   # 产物在 build/riscv64-linux-gnu/
#
# 用法（guest 内 native 编译）：
#   make
#   make test          编译 + 跑全部回归测试（SIM 模式，无需驱动）
#
# 用法（host 交叉编译完，guest 端只跑测试）：
#   guest$ make test   make 检测产物已是 RISC-V 二进制，直接跑
#
# 其它：
#   make install   安装到 /usr/local/lib + /usr/local/include/conflux
#   make clean     清理所有 build/ 目录

INCLUDE_DIR = include
SRC_DIR     = src
TEST_DIR    = tests
DRIVER_INC  = ../driver

# ---- 工具链 ----
# CROSS_COMPILE 是 Linux 内核 / U-Boot 通用约定。
# 默认空 → native；设置 'riscv64-linux-gnu-' 就是给 RISC-V 交叉编译。
CROSS_COMPILE ?=
CC := $(CROSS_COMPILE)gcc
AR := $(CROSS_COMPILE)ar

# ---- 按目标 triple 分开 build 目录 ----
# 这样 host 上 native 和 cross 的产物互不污染；
# guest（RISC-V）做 native 时，能直接复用 host 交叉编译的产物（同 triple）。
TARGET_TRIPLE := $(shell $(CC) -dumpmachine 2>/dev/null)
HOST_TRIPLE   := $(shell gcc -dumpmachine 2>/dev/null)
BUILD_DIR     := build/$(TARGET_TRIPLE)

# 当目标 triple 和 host triple 一致 → 可以直接跑测试
ifeq ($(TARGET_TRIPLE),$(HOST_TRIPLE))
CAN_RUN := 1
else
CAN_RUN := 0
endif

PREFIX  ?= /usr/local

CFLAGS  = -I$(INCLUDE_DIR) -I$(DRIVER_INC) -DHAVE_GPGPU_IOCTL_H \
          -D_GNU_SOURCE \
          -Wall -Wno-unused-parameter -fPIC -O2
LDFLAGS = -lpthread

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

LIB_A   = $(BUILD_DIR)/libconflux.a
LIB_SO  = $(BUILD_DIR)/libconflux.so

# 功能/回归测试：要求输出 "ALL PASSED"
TEST_SRCS = $(filter-out $(TEST_DIR)/test_perf%.c \
                          $(TEST_DIR)/test_ocl%.c \
                          $(TEST_DIR)/test_dma.c, \
                         $(wildcard $(TEST_DIR)/*.c))
TEST_BINS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))

# 性能测试：单独 target，不参与回归判定
PERF_SRCS = $(wildcard $(TEST_DIR)/test_perf*.c)
PERF_BINS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(PERF_SRCS))

# 设备集成测试（需要真实驱动，在 guest 上跑）
DMA_TEST_BIN = $(BUILD_DIR)/test_dma
OCL_TEST_BIN = $(BUILD_DIR)/test_ocl_mem

all: $(LIB_A) $(LIB_SO)
	@echo "  built  $(BUILD_DIR)  (target=$(TARGET_TRIPLE))"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	@echo "  CC     $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(LIB_A): $(OBJS)
	@echo "  AR     $(notdir $@)"
	@$(AR) rcs $@ $^

$(LIB_SO): $(OBJS)
	@echo "  LD     $(notdir $@)"
	@$(CC) -shared $^ -o $@ $(LDFLAGS)

# === 测试 ===
$(BUILD_DIR)/%: $(TEST_DIR)/%.c $(LIB_A)
	@mkdir -p $(BUILD_DIR)
	@echo "  TEST   $(notdir $@)"
	@$(CC) $(CFLAGS) $< $(LIB_A) -o $@ $(LDFLAGS)

# 编译测试可执行文件，但不一定跑（取决于架构是否匹配）
test-build: $(TEST_BINS)
	@echo "  built tests in $(BUILD_DIR)"

# 编译 + 跑测试。交叉编译时跳过运行，并提示去 guest 跑
test: $(TEST_BINS)
ifeq ($(CAN_RUN),1)
	@PASS=0; FAIL=0; \
	for t in $(TEST_BINS); do \
		name=$$(basename $$t); \
		if timeout 30 $$t > $(BUILD_DIR)/$$name.out 2>&1 && \
		   grep -qiE "all (tests |phases |quality tests )?passed" $(BUILD_DIR)/$$name.out; then \
			echo "  PASS   $$name"; \
			PASS=$$((PASS+1)); \
		else \
			echo "  FAIL   $$name"; \
			FAIL=$$((FAIL+1)); \
		fi; \
	done; \
	echo "--- $$PASS / $$((PASS+FAIL)) tests passing ---"; \
	test $$FAIL -eq 0
else
	@echo "  cross build ($(TARGET_TRIPLE) on $(HOST_TRIPLE))"
	@echo "  run tests on the target:"
	@echo "    cd $(CURDIR) && make test-run"
endif

# 仅运行测试，不重新编译。给交叉编译完后在 guest 上跑用。
test-run:
	@PASS=0; FAIL=0; \
	for t in $(TEST_BINS); do \
		name=$$(basename $$t); \
		if [ ! -x $$t ]; then echo "  MISS   $$name (not built)"; FAIL=$$((FAIL+1)); continue; fi; \
		if timeout 30 $$t > $(BUILD_DIR)/$$name.out 2>&1 && \
		   grep -qiE "all (tests |phases |quality tests )?passed" $(BUILD_DIR)/$$name.out; then \
			echo "  PASS   $$name"; \
			PASS=$$((PASS+1)); \
		else \
			echo "  FAIL   $$name"; \
			FAIL=$$((FAIL+1)); \
		fi; \
	done; \
	echo "--- $$PASS / $$((PASS+FAIL)) tests passing ---"; \
	test $$FAIL -eq 0

# 性能测试（手动跑，不参与回归）
perf: $(PERF_BINS)
ifeq ($(CAN_RUN),1)
	@for t in $(PERF_BINS); do echo "=== $$(basename $$t) ==="; $$t; done
else
	@echo "perf: cross build, run on target"
endif

# 设备集成测试：需要真实驱动 + /dev/gpgpu0，在 guest 上手动跑
$(DMA_TEST_BIN): $(TEST_DIR)/test_dma.c $(LIB_A)
	@mkdir -p $(BUILD_DIR)
	@echo "  TEST   test_dma"
	@$(CC) $(CFLAGS) $< $(LIB_A) -o $@ $(LDFLAGS)

test-device: $(DMA_TEST_BIN)
	@echo "  built $(DMA_TEST_BIN)"
	@echo "  run on guest:  LD_LIBRARY_PATH=$(BUILD_DIR) $(DMA_TEST_BIN)"

test-device-run: $(DMA_TEST_BIN)
	LD_LIBRARY_PATH=$(BUILD_DIR) $(DMA_TEST_BIN)

# OpenCL 集成测试：需要 PoCL 已安装（/usr/local/lib/libpocl.so + ICD loader）
# 在 guest 上运行：make test-ocl-run
OCL_CFLAGS  = $(CFLAGS) -I/usr/local/include
OCL_LDFLAGS = -lOpenCL -Wl,-rpath,/usr/local/lib

KERNEL_LAUNCH_BIN = $(BUILD_DIR)/test_kernel_launch

$(OCL_TEST_BIN): $(TEST_DIR)/test_ocl_mem.c
	@mkdir -p $(BUILD_DIR)
	@echo "  TEST   test_ocl_mem"
	@$(CC) $(OCL_CFLAGS) $< -o $@ $(OCL_LDFLAGS)

$(KERNEL_LAUNCH_BIN): $(TEST_DIR)/test_kernel_launch.c
	@mkdir -p $(BUILD_DIR)
	@echo "  TEST   test_kernel_launch"
	@$(CC) $(OCL_CFLAGS) $< -o $@ $(OCL_LDFLAGS)

test-ocl: $(OCL_TEST_BIN)
	@echo "  built $(OCL_TEST_BIN)"

test-ocl-run: $(OCL_TEST_BIN)
	POCL_DEVICES=gpgpu $(OCL_TEST_BIN)

test-kernel: $(KERNEL_LAUNCH_BIN)
	@echo "  built $(KERNEL_LAUNCH_BIN)"

test-kernel-run: $(KERNEL_LAUNCH_BIN)
	POCL_DEVICES=gpgpu $(KERNEL_LAUNCH_BIN)

# === Sanitizer 构建（用于在 guest 上检测内存/线程问题）===
# ASAN：检测 use-after-free、越界、泄漏
#   make test-asan        编译 + 跑 SIM 回归测试（不需要驱动）
#   make test-dma-asan    编译 + 跑 DMA 集成测试（需要 /dev/gpgpu0）
#
# 注意：TSAN（线程分析）目前 RISC-V 工具链没有 libtsan，暂不支持。
#       如需 TSAN，在 x86 host 上用 native 编译的 mock 版本跑。

ASAN_DIR     = $(BUILD_DIR)-asan
ASAN_FLAGS   = -fsanitize=address -fno-omit-frame-pointer -g -O1
ASAN_SRCS    = $(SRCS)
ASAN_OBJS    = $(patsubst $(SRC_DIR)/%.c,$(ASAN_DIR)/%.o,$(ASAN_SRCS))
ASAN_LIB_A   = $(ASAN_DIR)/libconflux.a
ASAN_TEST_BINS = $(patsubst $(TEST_DIR)/%.c,$(ASAN_DIR)/%,$(TEST_SRCS))
ASAN_DMA_BIN   = $(ASAN_DIR)/test_dma

$(ASAN_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(ASAN_DIR)
	@echo "  CC[asan] $<"
	@$(CC) $(CFLAGS) $(ASAN_FLAGS) -c $< -o $@

$(ASAN_LIB_A): $(ASAN_OBJS)
	@echo "  AR[asan] $(notdir $@)"
	@$(AR) rcs $@ $^

$(ASAN_DIR)/%: $(TEST_DIR)/%.c $(ASAN_LIB_A)
	@mkdir -p $(ASAN_DIR)
	@echo "  TEST[asan] $(notdir $@)"
	@$(CC) $(CFLAGS) $(ASAN_FLAGS) $< $(ASAN_LIB_A) -o $@ $(LDFLAGS)

$(ASAN_DMA_BIN): $(TEST_DIR)/test_dma.c $(ASAN_LIB_A)
	@mkdir -p $(ASAN_DIR)
	@echo "  TEST[asan] test_dma"
	@$(CC) $(CFLAGS) $(ASAN_FLAGS) $< $(ASAN_LIB_A) -o $@ $(LDFLAGS)

test-asan: $(ASAN_TEST_BINS)
ifeq ($(CAN_RUN),1)
	@PASS=0; FAIL=0; \
	for t in $(ASAN_TEST_BINS); do \
		name=$$(basename $$t); \
		if ASAN_OPTIONS=detect_leaks=1 timeout 30 $$t > $(ASAN_DIR)/$$name.out 2>&1 && \
		   grep -qiE "all (tests |phases |quality tests )?passed" $(ASAN_DIR)/$$name.out; then \
			echo "  PASS   $$name (asan)"; \
			PASS=$$((PASS+1)); \
		else \
			echo "  FAIL   $$name (asan)"; \
			cat $(ASAN_DIR)/$$name.out; \
			FAIL=$$((FAIL+1)); \
		fi; \
	done; \
	echo "--- $$PASS / $$((PASS+FAIL)) tests passing (asan) ---"; \
	test $$FAIL -eq 0
else
	@echo "  cross build — run on guest: make test-asan-run"
endif

test-asan-run:
	@PASS=0; FAIL=0; \
	for t in $(ASAN_TEST_BINS); do \
		name=$$(basename $$t); \
		[ -x $$t ] || { echo "  MISS   $$name"; FAIL=$$((FAIL+1)); continue; }; \
		if ASAN_OPTIONS=detect_leaks=1 timeout 30 $$t > $(ASAN_DIR)/$$name.out 2>&1 && \
		   grep -qiE "all (tests |phases |quality tests )?passed" $(ASAN_DIR)/$$name.out; then \
			echo "  PASS   $$name (asan)"; \
			PASS=$$((PASS+1)); \
		else \
			echo "  FAIL   $$name (asan)"; \
			cat $(ASAN_DIR)/$$name.out; \
			FAIL=$$((FAIL+1)); \
		fi; \
	done; \
	echo "--- $$PASS / $$((PASS+FAIL)) tests passing (asan) ---"; \
	test $$FAIL -eq 0

test-dma-asan: $(ASAN_DMA_BIN)
	ASAN_OPTIONS=detect_leaks=1 $(ASAN_DMA_BIN)

# === 安装 ===
install: $(LIB_A) $(LIB_SO)
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include/conflux
	install -m 644 $(LIB_A) $(LIB_SO) $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(INCLUDE_DIR)/*.h $(DESTDIR)$(PREFIX)/include/conflux/
	@echo "Installed libconflux to $(PREFIX) (target=$(TARGET_TRIPLE))"

clean:
	@rm -rf build
	@echo "  CLEAN  conflux-runtime (all triples)"

.PHONY: all test test-build test-run perf test-device test-device-run \
        test-ocl test-ocl-run test-kernel test-kernel-run \
        test-asan test-asan-run test-dma-asan \
        install clean
