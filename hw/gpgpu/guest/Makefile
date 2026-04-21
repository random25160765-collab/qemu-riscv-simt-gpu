# GPGPU Project Makefile
# ============================================================

# 目录结构
DRIVER_DIR = driver
RUNTIME_DIR = runtime
KERNEL_DIR = kernels
TEST_DIR = tests
BIN_DIR = bin

# 工具链
RISCV_PREFIX = riscv64-linux-gnu-
RISCV_CFLAGS = -march=rv32g -mabi=ilp32 -O2 -nostdlib -ffreestanding
RISCV_LDFLAGS = -T $(KERNEL_DIR)/kernel.ld

# 主机编译器
HOST_CC = gcc
HOST_CFLAGS = -Wall -O2 -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
HOST_LDFLAGS = -lm

# 自动发现所有 kernel 汇编文件
KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.S)
KERNEL_BINS = $(patsubst $(KERNEL_DIR)/%.S,$(BIN_DIR)/kernels/%.bin,$(KERNEL_SRCS))

# 自动发现所有测试程序（排除 cuda_* 系列，单独处理）
TEST_SRCS = $(filter-out $(TEST_DIR)/test_cuda_%.c,$(wildcard $(TEST_DIR)/test_*.c))
TEST_BINS = $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SRCS))
TEST_NAMES = $(patsubst $(TEST_DIR)/test_%.c,%,$(TEST_SRCS))

# CUDA-like API 测试（每个函数独立测试，依赖全部 kernel）
CUDA_TEST_SRCS = $(wildcard $(TEST_DIR)/test_cuda_*.c)
CUDA_TEST_NAMES = $(patsubst $(TEST_DIR)/test_cuda_%.c,%,$(CUDA_TEST_SRCS))

# 运行时库
RUNTIME_LIB = $(BIN_DIR)/libgpgpu.a

# 驱动模块
DRIVER_KO = $(BIN_DIR)/driver/gpgpu.ko

# ============================================================
# 默认目标
# ============================================================
.PHONY: all
all: driver kernels runtime
	@echo ""
	@echo "=== Build Complete ==="
	@echo "Run 'make install' to load the driver"
	@echo "Run 'make test' to run all tests"
	@echo "Run 'make test-<name>' to run a specific test"
	@echo "Run 'make test-cuda' to run all CUDA-like API tests"
	@echo "Run 'make test-cuda-<name>' to run a specific CUDA API test"
	@echo "Available tests: $(TEST_NAMES)"
	@echo "Available CUDA tests: $(CUDA_TEST_NAMES)"

# ============================================================
# 驱动编译
# ============================================================
.PHONY: driver
driver: $(DRIVER_KO)

$(DRIVER_KO): $(DRIVER_DIR)/gpgpu.c $(DRIVER_DIR)/gpgpu.h $(DRIVER_DIR)/gpgpu_ioctl.h
	@mkdir -p $(BIN_DIR)/driver
	@echo "  CC       gpgpu.ko"
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) modules
	@mv $(DRIVER_DIR)/gpgpu.ko $(BIN_DIR)/driver/
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) clean

.PHONY: install
install: $(DRIVER_KO)
	@sudo insmod $(DRIVER_KO)
	@echo "Driver loaded."

.PHONY: remove
remove:
	@sudo rmmod gpgpu 2>/dev/null || true
	@echo "Driver removed."

.PHONY: reload
reload: remove install

# ============================================================
# Kernel 编译（自动发现所有 .S 文件）
# ============================================================
.PHONY: kernels
kernels: $(KERNEL_BINS)

$(BIN_DIR)/kernels/%.bin: $(KERNEL_DIR)/%.S $(KERNEL_DIR)/kernel.ld
	@mkdir -p $(BIN_DIR)/kernels
	@echo "  KERNEL   $*"
	@$(RISCV_PREFIX)gcc -c $(RISCV_CFLAGS) $< -o $(BIN_DIR)/kernels/$*.o
	@$(RISCV_PREFIX)ld $(RISCV_LDFLAGS) $(BIN_DIR)/kernels/$*.o -o $(BIN_DIR)/kernels/$*.elf
	@$(RISCV_PREFIX)objcopy -O binary $(BIN_DIR)/kernels/$*.elf $@

.PHONY: clean-kernels
clean-kernels:
	@rm -rf $(BIN_DIR)/kernels

# ============================================================
# 运行时库
# ============================================================
.PHONY: runtime
runtime: $(RUNTIME_LIB)

$(RUNTIME_LIB): $(RUNTIME_DIR)/gpgpu_runtime.c $(RUNTIME_DIR)/gpgpu_runtime.h
	@mkdir -p $(BIN_DIR)
	@echo "  LIB      libgpgpu.a"
	@$(HOST_CC) -c $(HOST_CFLAGS) $< -o $(BIN_DIR)/gpgpu_runtime.o
	@ar rcs $@ $(BIN_DIR)/gpgpu_runtime.o

# ============================================================
# 测试程序（自动发现 test_*.c）
# ============================================================
.PHONY: test
test: $(patsubst %,test-%,$(TEST_NAMES))
	@echo ""
	@echo "=== All Tests Complete ==="

# 生成每个测试的规则
define make_test_rule
.PHONY: test-$(1)
test-$(1): $(RUNTIME_LIB) $(BIN_DIR)/kernels/$(1).bin
	@echo "  TEST     $(1)"
	@$(HOST_CC) $(HOST_CFLAGS) $(TEST_DIR)/test_$(1).c -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $(BIN_DIR)/test_$(1)
	@sudo $(BIN_DIR)/test_$(1)
endef

$(foreach t,$(filter-out cnn,$(TEST_NAMES)),$(eval $(call make_test_rule,$(t))))

# CNN 测试需要所有 kernels
.PHONY: test-cnn
test-cnn: $(RUNTIME_LIB) $(KERNEL_BINS)
	@echo "  TEST     cnn"
	@$(HOST_CC) $(HOST_CFLAGS) $(TEST_DIR)/test_cnn.c -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $(BIN_DIR)/test_cnn
	@sudo $(BIN_DIR)/test_cnn

# ============================================================
# CUDA-like API 测试（每个测试依赖全部 kernel）
# ============================================================
.PHONY: test-cuda
test-cuda: $(patsubst %,test-cuda-%,$(CUDA_TEST_NAMES))
	@echo ""
	@echo "=== All CUDA API Tests Complete ==="

define make_cuda_test_rule
.PHONY: test-cuda-$(1)
test-cuda-$(1): $(RUNTIME_LIB) $(KERNEL_BINS)
	@echo "  TEST     cuda/$(1)"
	@$(HOST_CC) $(HOST_CFLAGS) $(TEST_DIR)/test_cuda_$(1).c -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $(BIN_DIR)/test_cuda_$(1)
	@sudo $(BIN_DIR)/test_cuda_$(1)
endef

$(foreach t,$(CUDA_TEST_NAMES),$(eval $(call make_cuda_test_rule,$(t))))

# 编译所有 CUDA 测试（不运行）
.PHONY: build-cuda-tests
build-cuda-tests: $(RUNTIME_LIB) $(KERNEL_BINS)
	@$(foreach t,$(CUDA_TEST_NAMES), \
		echo "  CC       cuda/$(t)" && \
		$(HOST_CC) $(HOST_CFLAGS) $(TEST_DIR)/test_cuda_$(t).c \
			-L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $(BIN_DIR)/test_cuda_$(t) &&) true

# ============================================================
# 编译单个测试（不运行）
# ============================================================
$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.c $(RUNTIME_LIB)
	@mkdir -p $(BIN_DIR)
	@echo "  CC       $*"
	@$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

.PHONY: build-tests
build-tests: $(TEST_BINS)

# ============================================================
# 示例程序
# ============================================================
.PHONY: cnn
cnn: $(RUNTIME_LIB) $(KERNEL_BINS)
	@echo "  EXAMPLE  cnn"
	@$(HOST_CC) $(HOST_CFLAGS) cuda_cnn.c -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $(BIN_DIR)/cnn
	@sudo $(BIN_DIR)/cnn

# ============================================================
# 清理
# ============================================================
.PHONY: clean
clean:
	@rm -rf $(BIN_DIR)
	@echo "Cleaned."

# ============================================================
# 调试
# ============================================================
.PHONY: dmesg
dmesg:
	@sudo dmesg -w | grep -E "gpgpu|KERNEL|DEVICE"

.PHONY: dmesg-clear
dmesg-clear:
	@sudo dmesg -c > /dev/null

.PHONY: status
status:
	@echo "=== GPGPU Status ==="
	@ls -la /dev/gpgpu* 2>/dev/null || echo "No device nodes"
	@lsmod | grep gpgpu || echo "Driver not loaded"
	@lspci -d 1234:1337 2>/dev/null || echo "Device not found"

# ============================================================
# Python 绑定（pybind11）
# ============================================================
#
# 构建方式：
#   make python
#
# 前置依赖：
#   sudo apt install pybind11-dev python3-dev
#   pip3 install numpy
#
# 运行测试（需要 root 访问 /dev/gpgpu0）：
#   cd python
#   export GPGPU_KERNEL_DIR=$(pwd)/../bin/kernels
#   sudo python3 test_gpgpu.py

PYTHON_DIR = python
PYTHON     ?= python3

PY_CFLAGS  = $(shell $(PYTHON)-config --includes 2>/dev/null)
PY_SUFFIX  = $(shell $(PYTHON)-config --extension-suffix 2>/dev/null)
PY_EXT     = $(PYTHON_DIR)/gpgpu$(PY_SUFFIX)

PYBIND11_INC ?= $(shell $(PYTHON) -c \
    "import pybind11; print(pybind11.get_include())" 2>/dev/null || \
    echo "/usr/include/pybind11")

NUMPY_INC ?= $(shell $(PYTHON) -c \
    "import numpy; print(numpy.get_include())" 2>/dev/null || \
    echo "")

CXX ?= g++
PY_CXXFLAGS = -std=c++17 -O2 -Wall -fPIC -shared \
              -I$(RUNTIME_DIR) -I$(DRIVER_DIR) \
              $(PY_CFLAGS) \
              -I$(PYBIND11_INC) \
              $(if $(NUMPY_INC),-I$(NUMPY_INC),)
PY_LDFLAGS = -L$(BIN_DIR) -lgpgpu -lm

.PHONY: python
python: $(RUNTIME_LIB)
	@mkdir -p $(PYTHON_DIR)
	@echo "  PY       $(PY_EXT)"
	@$(CXX) $(PY_CXXFLAGS) \
		$(PYTHON_DIR)/gpgpu_module.cpp \
		$(PY_LDFLAGS) \
		-o $(PY_EXT)
	@echo ""
	@echo "=== Python 模块构建完成 ==="
	@echo "模块路径：$(PY_EXT)"
	@echo "使用方式："
	@echo "  cd $(PYTHON_DIR)"
	@echo "  export GPGPU_KERNEL_DIR=\$$(pwd)/../bin/kernels"
	@echo "  sudo python3 test_gpgpu.py"

.PHONY: clean-python
clean-python:
	@rm -f $(PYTHON_DIR)/gpgpu*.so
	@echo "Python 模块已清理。"

# ============================================================
# 帮助
# ============================================================
.PHONY: help
help:
	@echo "GPGPU Project Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build driver, kernels, and runtime"
	@echo "  install       - Load the driver"
	@echo "  remove        - Unload the driver"
	@echo "  reload        - Reload the driver"
	@echo "  test          - Run all tests"
	@echo "  test-<name>   - Run specific test"
	@echo "  test-cuda     - Run all CUDA-like API tests"
	@echo "  test-cuda-<n> - Run specific CUDA API test"
	@echo "  cnn           - Run CNN example"
	@echo "  python        - Build Python bindings (requires pybind11-dev)"
	@echo "  clean         - Clean build artifacts"
	@echo "  clean-python  - Clean Python .so"
	@echo "  status        - Show device status"
	@echo "  dmesg         - Watch kernel logs"
	@echo ""
	@echo "Available tests: $(TEST_NAMES)"
	@echo "Available CUDA tests: $(CUDA_TEST_NAMES)"
	@echo "Available kernels: $(notdir $(basename $(KERNEL_SRCS)))"