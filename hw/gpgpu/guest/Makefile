# GPGPU Project Top Makefile
# ============================================================

DRIVER_DIR = driver
RUNTIME_DIR = runtime
KERNEL_DIR = kernel
PYTHON_DIR = python
BIN_DIR = bin

KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.S)
KERNEL_NAMES = $(basename $(notdir $(KERNEL_SRCS)))
KERNEL_BINS = $(patsubst $(KERNEL_DIR)/%.S,$(BIN_DIR)/kernel/%.bin,$(KERNEL_SRCS))

RISCV_PREFIX = riscv64-linux-gnu-
RISCV_CFLAGS = -march=rv32g -mabi=ilp32 -O2 -nostdlib -ffreestanding
RISCV_LDFLAGS = -T $(KERNEL_DIR)/kernel.ld

HOST_CC = gcc
HOST_CFLAGS = -Wall -O2 -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
HOST_LDFLAGS = -lm

# Python binding
PYTHON     ?= python3
PY_CFLAGS  = $(shell $(PYTHON)-config --includes 2>/dev/null)
PY_SUFFIX  = $(shell $(PYTHON)-config --extension-suffix 2>/dev/null)
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

DRIVER_KO = $(BIN_DIR)/driver/gpgpu.ko
RUNTIME_LIB = $(BIN_DIR)/libgpgpu.a
PY_EXT     = $(PYTHON_DIR)/gpgpu$(PY_SUFFIX)

ifneq ($(shell which ccache 2>/dev/null),)
    HOST_CC = ccache gcc
    CXX = ccache g++
endif

# ============================= Compile Rules ===============================

driver: $(DRIVER_KO)
$(DRIVER_KO): $(DRIVER_DIR)/gpgpu.c $(DRIVER_DIR)/gpgpu.h $(DRIVER_DIR)/gpgpu_ioctl.h
	@mkdir -p $(BIN_DIR)/driver
	@echo "  CC       gpgpu.ko"
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) modules
	@mv $(DRIVER_DIR)/gpgpu.ko $(BIN_DIR)/driver/
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) clean

kernel: $(KERNEL_BINS)
$(BIN_DIR)/kernel/%.bin: $(KERNEL_DIR)/%.S $(KERNEL_DIR)/kernel.ld
	@mkdir -p $(BIN_DIR)/kernel
	@echo "  KERNEL   $*"
	$(RISCV_PREFIX)gcc -c $(RISCV_CFLAGS) $< -o $(BIN_DIR)/kernel/$*.o
	$(RISCV_PREFIX)ld $(RISCV_LDFLAGS) $(BIN_DIR)/kernel/$*.o -o $(BIN_DIR)/kernel/$*.elf
	$(RISCV_PREFIX)objcopy -O binary $(BIN_DIR)/kernel/$*.elf $@

runtime: $(RUNTIME_LIB)
$(RUNTIME_LIB): $(RUNTIME_DIR)/gpgpu_runtime.c $(RUNTIME_DIR)/gpgpu_runtime.h
	@mkdir -p $(BIN_DIR)
	@echo "  LIB      libgpgpu.a"
	$(HOST_CC) -c $(HOST_CFLAGS) $< -o $(BIN_DIR)/gpgpu_runtime.o
	ar rcs $@ $(BIN_DIR)/gpgpu_runtime.o

python: $(RUNTIME_LIB)
	@mkdir -p $(PYTHON_DIR)
	@echo "  PY       $(PY_EXT)"
	$(CXX) $(PY_CXXFLAGS) \
		$(PYTHON_DIR)/gpgpu_module.cpp \
		$(PY_LDFLAGS) \
		-o $(PY_EXT)

# ============================= Clean Rules ===============================

clean: clean-driver clean-kernel clean-runtime clean-python
	@echo "All Cleaned."

clean-driver:
	@rm -rf $(BIN_DIR)/driver
	@echo "Driver cleaned."

clean-kernel:
	@rm -rf $(BIN_DIR)/kernel
	@echo "Kernel cleaned."

clean-runtime:
	@rm -f $(RUNTIME_LIB) $(BIN_DIR)/gpgpu_runtime.o
	@echo "Runtime cleaned."

clean-python:
	@rm -f $(PYTHON_DIR)/gpgpu*.so
	@echo "Python cleaned."

.PHONY: driver kernel runtime python
.PHONY: clean clean-driver clean-kernel clean-runtime clean-python
