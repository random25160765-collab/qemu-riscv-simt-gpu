# GPGPU Project Top Makefile
# ============================================================

DRIVER_DIR = driver
RUNTIME_DIR = runtime
KERNEL_DIR = kernel
PYTHON_DIR = python
BIN_DIR = bin

export GPGPU_DEVICE = /dev/gpgpu0
export GPGPU_KERNEL_DIR = $(BIN_DIR)/kernel

include scripts/toolchain.mk
include scripts/control.mk
include scripts/test.mk
include scripts/common.mk

RISCV_PREFIX = riscv64-linux-gnu-
RISCV_CC = $(RISCV_PREFIX)gcc
RISCV_CXX = $(RISCV_PREFIX)g++
RISCV_AR = $(RISCV_PREFIX)ar
RISCV_LD = $(RISCV_PREFIX)ld
RISCV_OBJCOPY = $(RISCV_PREFIX)objcopy
RISCV_STRIP = $(RISCV_PREFIX)strip

RISCV_CFLAGS_KERNEL = -march=rv32g -mabi=ilp32 -O2 -nostdlib -ffreestanding
RISCV_LDFLAGS_KERNEL = -T $(KERNEL_DIR)/kernel.ld

RISCV_CFLAGS_RUNTIME = -march=rv64gc -mabi=lp64d -O2 -Wall -fPIC
RISCV_CXXFLAGS_RUNTIME = -std=c++17 -O2 -Wall -fPIC -Wno-attributes

# Python toolchain
PROJECT_ROOT := $(shell pwd)
RISCV_TOOLCHAIN_DIR = $(PROJECT_ROOT)/toolchain
RISCV_PYTHON_INC = $(RISCV_TOOLCHAIN_DIR)/include/python3.13
RISCV_PYTHON_LIB = $(RISCV_TOOLCHAIN_DIR)/lib

HOST_CC = gcc
HOST_CFLAGS = -Wall -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
HOST_LDFLAGS = -lm

RISCV_PY_CXXFLAGS = $(RISCV_CXXFLAGS_RUNTIME) -shared \
                    -I$(RISCV_PYTHON_INC) \
                    -I$(RISCV_TOOLCHAIN_DIR)/include \
                    -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
RISCV_PY_LDFLAGS = -L$(BIN_DIR) -L$(RISCV_PYTHON_LIB) -lgpgpu -lpython3.13

KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.S)
KERNEL_NAMES = $(basename $(notdir $(KERNEL_SRCS)))
KERNEL_BINS = $(patsubst $(KERNEL_DIR)/%.S,$(BIN_DIR)/kernel/%.bin,$(KERNEL_SRCS))

DRIVER_KO = $(BIN_DIR)/driver/gpgpu.ko
RUNTIME_LIB = $(BIN_DIR)/libgpgpu.a
RISCV_PY_EXT = $(BIN_DIR)/python/gpgpu_riscv.so

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
	$(RISCV_CC) -c $(RISCV_CFLAGS_KERNEL) $< -o $(BIN_DIR)/kernel/$*.o
	$(RISCV_LD) $(RISCV_LDFLAGS_KERNEL) $(BIN_DIR)/kernel/$*.o -o $(BIN_DIR)/kernel/$*.elf
	$(RISCV_OBJCOPY) -O binary $(BIN_DIR)/kernel/$*.elf $@
	@rm -f $(BIN_DIR)/kernel/$*.o $(BIN_DIR)/kernel/$*.elf

runtime: $(RUNTIME_LIB)
$(RUNTIME_LIB): $(RUNTIME_DIR)/gpgpu_runtime.c $(RUNTIME_DIR)/gpgpu_runtime.h
	@mkdir -p $(BIN_DIR)
	@echo "  LIB      libgpgpu.a"
	$(RISCV_CC) -c $(RISCV_CFLAGS_RUNTIME) -I$(DRIVER_DIR) $< -o $(BIN_DIR)/gpgpu_runtime.o
	$(RISCV_AR) rcs $@ $(BIN_DIR)/gpgpu_runtime.o

python: $(RUNTIME_LIB) $(RISCV_PY_EXT)
$(RISCV_PY_EXT): $(PYTHON_DIR)/gpgpu_module.cpp
	@mkdir -p $(BIN_DIR)/python
	@echo "  RISCV    $(notdir $@)"
	@if [ ! -f "$(RISCV_PYTHON_INC)/Python.h" ]; then \
		echo "Error: RISC-V Python toolchain not found at $(RISCV_TOOLCHAIN_DIR)"; \
		echo "Please run: make setup-toolchain"; \
		exit 1; \
	fi
	$(RISCV_CXX) $(RISCV_PY_CXXFLAGS) $< $(RISCV_PY_LDFLAGS) -o $@
	@file $@

# ============================= Clean Rules ===============================

clean: clean-driver clean-kernel clean-runtime clean-python
	@rm -rf $(BIN_DIR)
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
	@rm -rf $(BIN_DIR)/python
	@echo "Python cleaned."

.PHONY: driver kernel runtime python
.PHONY: clean clean-driver clean-kernel clean-runtime clean-python
