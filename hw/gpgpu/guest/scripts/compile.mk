# GPGPU Project Compile Makefile
# ============================================================

S = @

# ============================================================
# =                           MAIN                           =
# ============================================================

RISCV_PREFIX = riscv64-linux-gnu-
RISCV_CC = $(RISCV_PREFIX)gcc
RISCV_CXX = $(RISCV_PREFIX)g++
RISCV_AR = $(RISCV_PREFIX)ar
RISCV_OBJCOPY = $(RISCV_PREFIX)objcopy
RISCV_STRIP = $(RISCV_PREFIX)strip

RISCV_CFLAGS_KERNEL = -march=rv32g -mabi=ilp32 -O2 -nostdlib -ffreestanding
RISCV_LDFLAGS_KERNEL = -nostdlib -T $(KERNEL_DIR)/kernel.ld

RISCV_CFLAGS_RUNTIME = -march=rv64gc -mabi=lp64d -O2 -Wall -fPIC

RISCV_PY_CXXFLAGS = -std=c++17 -O2 -Wall -fPIC -Wno-attributes -shared \
                    -I$(RISCV_PYTHON_INC) \
                    -I$(TOOLCHAIN_DIR)/include \
                    -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
RISCV_PY_LDFLAGS = -L$(BIN_DIR) -L$(RISCV_PYTHON_LIB) -lgpgpu -lpython3.13

KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.S)
KERNEL_NAMES = $(basename $(notdir $(KERNEL_SRCS)))
KERNEL_BINS = $(patsubst $(KERNEL_DIR)/%.S,$(BIN_DIR)/kernel/%.bin,$(KERNEL_SRCS))

DRIVER_KO = $(BIN_DIR)/driver/gpgpu.ko
RUNTIME_LIB = $(BIN_DIR)/libgpgpu.a
RISCV_PY_EXT = $(BIN_DIR)/python/gpgpu_riscv.so

# ============================= Compile Rules ===============================

driver: $(DRIVER_KO)
$(DRIVER_KO): $(DRIVER_DIR)/gpgpu.c $(DRIVER_DIR)/gpgpu.h $(DRIVER_DIR)/gpgpu_ioctl.h
	@mkdir -p $(BIN_DIR)/driver
	@echo "  CC       gpgpu.ko"
	$(S) $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) modules
	@mv $(DRIVER_DIR)/gpgpu.ko $(BIN_DIR)/driver/
	$(S) $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) clean

kernel: $(KERNEL_BINS)
$(BIN_DIR)/kernel/%.bin: $(KERNEL_DIR)/%.S $(KERNEL_DIR)/kernel.ld
	@mkdir -p $(BIN_DIR)/kernel
	@echo "  KERNEL   $*"
	$(S) $(RISCV_CC) -c $(RISCV_CFLAGS_KERNEL) $< -o $(BIN_DIR)/kernel/$*.o
	$(S) $(RISCV_CC) $(RISCV_LDFLAGS_KERNEL) $(BIN_DIR)/kernel/$*.o -o $(BIN_DIR)/kernel/$*.elf
	$(S) $(RISCV_OBJCOPY) -O binary $(BIN_DIR)/kernel/$*.elf $@
	$(S) rm $(BIN_DIR)/kernel/$*.o $(BIN_DIR)/kernel/$*.elf

runtime: $(RUNTIME_LIB)
$(RUNTIME_LIB): $(RUNTIME_DIR)/gpgpu_runtime.c $(RUNTIME_DIR)/gpgpu_runtime.h
	@mkdir -p $(BIN_DIR)
	@echo "  LIB      libgpgpu.a"
	$(S) $(RISCV_CC) -c $(RISCV_CFLAGS_RUNTIME) -I$(DRIVER_DIR) $< -o $(BIN_DIR)/gpgpu_runtime.o
	$(S) $(RISCV_AR) rcs $@ $(BIN_DIR)/gpgpu_runtime.o
	$(S) rm $(BIN_DIR)/gpgpu_runtime.o

python: $(RUNTIME_LIB) $(RISCV_PY_EXT)
$(RISCV_PY_EXT): $(PYTHON_DIR)/gpgpu_module.cpp
	@mkdir -p $(BIN_DIR)/python
	@echo "  RISCV    $(notdir $@)"
	@if [ ! -f "$(RISCV_PYTHON_INC)/Python.h" ]; then \
		echo "Error: RISC-V Python toolchain not found at $(TOOLCHAIN_DIR)"; \
		echo "Please run: make setup-toolchain"; \
		exit 1; \
	fi
	$(S) $(RISCV_CXX) $(RISCV_PY_CXXFLAGS) $< $(RISCV_PY_LDFLAGS) -o $@
	@test -s $@ || { echo "  ERROR    $@ is empty or missing"; exit 1; }

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


# ============================================================
# =                           TEST                           =
# ============================================================

TEST_DIR = tests
TEST_DRIVER_DIR = $(TEST_DIR)/driver
TEST_KERNEL_DIR = $(TEST_DIR)/kernel
TEST_CUDA_DIR   = $(TEST_DIR)/cuda
TEST_PYTHON_DIR = $(TEST_DIR)/python

TEST_DRIVER_SRCS  = $(wildcard $(TEST_DRIVER_DIR)/*.c)
TEST_DRIVER_NAMES = $(basename $(notdir $(TEST_DRIVER_SRCS)))
TEST_DRIVER_BINS  = $(patsubst $(TEST_DRIVER_DIR)/%.c,$(BIN_DIR)/tests/driver/%,$(TEST_DRIVER_SRCS))

TEST_KERNEL_SRCS  = $(wildcard $(TEST_KERNEL_DIR)/*.c)
TEST_KERNEL_NAMES = $(basename $(notdir $(TEST_KERNEL_SRCS)))
TEST_KERNEL_BINS  = $(patsubst $(TEST_KERNEL_DIR)/%.c,$(BIN_DIR)/tests/kernel/%,$(TEST_KERNEL_SRCS))

TEST_CUDA_SRCS    = $(wildcard $(TEST_CUDA_DIR)/*.c)
TEST_CUDA_NAMES   = $(basename $(notdir $(TEST_CUDA_SRCS)))
TEST_CUDA_BINS    = $(patsubst $(TEST_CUDA_DIR)/%.c,$(BIN_DIR)/tests/cuda/%,$(TEST_CUDA_SRCS))

TEST_PYTHON_SRCS  = $(wildcard $(TEST_PYTHON_DIR)/*.py)
TEST_PYTHON_NAMES = $(basename $(notdir $(TEST_PYTHON_SRCS)))

TEST_CFLAGS  = -march=rv64gc -mabi=lp64d -O2 -Wall -Wno-unused-result -fPIC -I$(RUNTIME_DIR) -I$(DRIVER_DIR)
TEST_LDFLAGS = -lm

# ============================ Compile Rules ================================

$(BIN_DIR)/tests/driver/%: $(TEST_DRIVER_DIR)/%.c $(RUNTIME_LIB)
	@mkdir -p $(BIN_DIR)/tests/driver
	@echo "  TEST     driver/$*"
	$(S) $(RISCV_CC) $(TEST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(TEST_LDFLAGS) -o $@

$(BIN_DIR)/tests/kernel/%: $(TEST_KERNEL_DIR)/%.c $(RUNTIME_LIB)
	@mkdir -p $(BIN_DIR)/tests/kernel
	@echo "  TEST     kernel/$*"
	$(S) $(RISCV_CC) $(TEST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(TEST_LDFLAGS) -o $@

$(BIN_DIR)/tests/cuda/%: $(TEST_CUDA_DIR)/%.c $(RUNTIME_LIB) $(KERNEL_BINS)
	@mkdir -p $(BIN_DIR)/tests/cuda
	@echo "  TEST     cuda/$*"
	$(S) $(RISCV_CC) $(TEST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(TEST_LDFLAGS) -o $@

# -------------------- Build All -------------------- #

build-driver-tests: $(TEST_DRIVER_BINS)
	@echo "Built driver tests: $(TEST_DRIVER_NAMES)"

build-kernel-tests: $(TEST_KERNEL_BINS)
	@echo "Built kernel tests: $(TEST_KERNEL_NAMES)"

build-cuda-tests: $(TEST_CUDA_BINS)
	@echo "Built CUDA tests: $(TEST_CUDA_NAMES)"

build-all-tests: build-driver-tests build-kernel-tests build-cuda-tests


# ============================ Clean Rules ================================

clean-driver-tests:
	@rm -rf $(BIN_DIR)/tests/driver
	@echo "Driver tests cleaned."

clean-kernel-tests:
	@rm -rf $(BIN_DIR)/tests/kernel
	@echo "Kernel tests cleaned."

clean-cuda-tests:
	@rm -rf $(BIN_DIR)/tests/cuda
	@echo "CUDA tests cleaned."

clean-python-tests:
	@rm -f $(BIN_DIR)/python/gpgpu*.so 2>/dev/null || true
	@rm -rf $(TEST_PYTHON_DIR)/__pycache__
	@rm -rf $(TEST_PYTHON_DIR)/*.pyc
	@echo "Python tests cleaned."

clean-tests:
	@rm -rf $(BIN_DIR)/tests
	@echo "Tests cleaned."


.PHONY: driver kernel runtime python
.PHONY: clean clean-driver clean-kernel clean-runtime clean-python

.PHONY: build-driver-tests build-kernel-tests build-cuda-tests build-all-tests
.PHONY: clean-tests
