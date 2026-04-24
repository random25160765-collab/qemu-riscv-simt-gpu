# GPGPU Project Test Makefile
# ============================================================

TEST_DIR = tests
TEST_DRIVER_DIR = $(TEST_DIR)/driver
TEST_KERNEL_DIR = $(TEST_DIR)/kernel
TEST_CUDA_DIR = $(TEST_DIR)/cuda
TEST_PYTHON_DIR = $(TEST_DIR)/python

# 二进制输出目录
TEST_BIN_DRIVER = $(BIN_DIR)/tests/driver
TEST_BIN_KERNEL = $(BIN_DIR)/tests/kernel
TEST_BIN_CUDA = $(BIN_DIR)/tests/cuda

# 源文件
TEST_DRIVER_SRCS = $(wildcard $(TEST_DRIVER_DIR)/*.c)
TEST_DRIVER_NAMES = $(patsubst test_%,%,$(basename $(notdir $(TEST_DRIVER_SRCS))))
TEST_DRIVER_BINS = $(addprefix $(TEST_BIN_DRIVER)/, $(TEST_DRIVER_NAMES))

TEST_KERNEL_SRCS = $(wildcard $(TEST_KERNEL_DIR)/*.c)
TEST_KERNEL_NAMES = $(basename $(notdir $(TEST_KERNEL_SRCS)))
TEST_KERNEL_BINS = $(addprefix $(TEST_BIN_KERNEL)/, $(TEST_KERNEL_NAMES))

TEST_CUDA_SRCS = $(wildcard $(TEST_CUDA_DIR)/*.c)
TEST_CUDA_NAMES = $(basename $(notdir $(TEST_CUDA_SRCS)))
TEST_CUDA_BINS = $(addprefix $(TEST_BIN_CUDA)/, $(TEST_CUDA_NAMES))

TEST_PYTHON_SRCS = $(wildcard $(TEST_PYTHON_DIR)/*.py)
TEST_PYTHON_NAMES = $(basename $(notdir $(TEST_PYTHON_SRCS)))

# ============================ Compile Rules ================================

$(TEST_BIN_DRIVER)/%: $(TEST_DRIVER_DIR)/test_%.c $(RUNTIME_LIB)
	@mkdir -p $(TEST_BIN_DRIVER)
	@echo "  TEST     driver/$*"
	$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

$(TEST_BIN_KERNEL)/%: $(TEST_KERNEL_DIR)/test_%.c $(RUNTIME_LIB)
	@mkdir -p $(TEST_BIN_KERNEL)
	@echo "  TEST     kernel/$*"
	$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

$(TEST_BIN_CUDA)/%: $(TEST_CUDA_DIR)/test_%.c $(RUNTIME_LIB) $(KERNEL_BINS)
	@mkdir -p $(TEST_BIN_CUDA)
	@echo "  TEST     cuda/$*"
	$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

# -------------------- Build All -------------------- #

build-driver-tests: $(TEST_DRIVER_BINS)
	@echo "Built driver tests: $(TEST_DRIVER_NAMES)"

build-kernel-tests: $(TEST_KERNEL_BINS)
	@echo "Built kernel tests: $(TEST_KERNEL_NAMES)"

build-cuda-tests: $(TEST_CUDA_BINS)
	@echo "Built CUDA tests: $(TEST_CUDA_NAMES)"

build-all-tests: build-driver-tests build-kernel-tests build-cuda-tests

# ============================ Run Rules ================================

# -------------------- Individual Tests -------------------- #

test-driver-%: $(TEST_BIN_DRIVER)/%
	@echo "  RUN      driver/$*"
	@sudo -E $<

test-kernel-%: $(TEST_BIN_KERNEL)/%
	@echo "  RUN      kernel/$*"
	@sudo -E $<

test-cuda-%: $(TEST_BIN_CUDA)/%
	@echo "  RUN      cuda/$*"
	@sudo -E $<

test-python-%: $(PYTHON_EXT)
	@echo "  RUN      python/$*"
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernel && \
		sudo -E $(PYTHON) $*.py

# -------------------- All Tests -------------------- #

test-driver-all: build-driver-tests
	@echo "=== Running all driver tests ==="
	@for test in $(TEST_DRIVER_NAMES); do \
		echo "  RUN      driver/$$test"; \
		sudo -E $(TEST_BIN_DRIVER)/$$test; \
	done

test-kernel-all: build-kernel-tests
	@echo "=== Running all kernel tests ==="
	@for test in $(TEST_KERNEL_NAMES); do \
		echo "  RUN      kernel/$$test"; \
		sudo -E $(TEST_BIN_KERNEL)/$$test; \
	done

test-cuda-all: build-cuda-tests
	@echo "=== Running all CUDA tests ==="
	@for test in $(TEST_CUDA_NAMES); do \
		echo "  RUN      cuda/$$test"; \
		sudo -E $(TEST_BIN_CUDA)/$$test; \
	done

test-python-all: $(PYTHON_EXT)
	@echo "=== Running all Python tests ==="
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernel && \
		for test in $(TEST_PYTHON_NAMES); do \
			echo "  RUN      python/$$test"; \
			sudo -E $(PYTHON) $$test.py; \
		done

# ============================ Quick Test ================================

.PHONY: build-tests
build-tests: build-driver-tests build-kernel-tests build-cuda-tests

.PHONY: test-all
test-all: test-driver-all test-kernel-all test-cuda-all test-python-all

# ============================ Clean Rules ================================

clean-driver-tests:
	@rm -rf $(TEST_BIN_DRIVER)
	@echo "Driver tests cleaned."

clean-kernel-tests:
	@rm -rf $(TEST_BIN_KERNEL)
	@echo "Kernel tests cleaned."

clean-cuda-tests:
	@rm -rf $(TEST_BIN_CUDA)
	@echo "CUDA tests cleaned."

clean-tests: clean-driver-tests clean-kernel-tests clean-cuda-tests
	@echo "All tests cleaned."

# ============================ Phony Declarations ================================

.PHONY: test-driver-% test-driver-all
.PHONY: test-kernel-% test-kernel-all
.PHONY: test-cuda-%   test-cuda-all
.PHONY: test-python-% test-python-all
.PHONY: build-driver-tests build-kernel-tests build-cuda-tests build-all-tests build-tests
.PHONY: test-all
.PHONY: clean-driver-tests clean-kernel-tests clean-cuda-tests clean-tests