# GPGPU Project Test Makefile
# ============================================================

TEST_DIR = $(BIN_DIR)/tests
TEST_DRIVER_DIR = $(TEST_DIR)/driver
TEST_KERNEL_DIR = $(TEST_DIR)/kernel
TEST_CUDA_DIR = $(TEST_DIR)/cuda
TEST_PYTHON_DIR = $(TEST_DIR)/python

TEST_DRIVER_SRCS = $(wildcard $(TEST_DRIVER_DIR)/*.c)
TEST_DRIVER_NAMES = $(basename $(notdir $(TEST_DRIVER_SRCS)))
TEST_DRIVER_BINS = $(patsubst $(TEST_DRIVER_DIR)/%.c,$(BIN_DIR)/%,$(TEST_DRIVER_SRCS))

TEST_KERNEL_SRCS = $(wildcard $(TEST_KERNEL_DIR)/*.c)
TEST_KERNEL_NAMES = $(basename $(notdir $(TEST_KERNEL_SRCS)))
TEST_KERNEL_BINS = $(patsubst $(TEST_KERNEL_DIR)/%.c,$(BIN_DIR)/%,$(TEST_KERNEL_SRCS))

TEST_CUDA_SRCS = $(wildcard $(TEST_CUDA_DIR)/*.c)
TEST_CUDA_NAMES = $(basename $(notdir $(TEST_CUDA_SRCS)))
TEST_CUDA_BINS = $(patsubst $(TEST_CUDA_DIR)/%.c,$(BIN_DIR)/%,$(TEST_CUDA_SRCS))

TEST_PYTHON_SRCS = $(wildcard $(TEST_PYTHON_DIR)/*.py)
TEST_PYTHON_NAMES = $(basename $(notdir $(TEST_PYTHON_SRCS)))

# ============================ Compile Rules ================================

$(BIN_DIR)/tests/driver/%: $(TEST_DRIVER_DIR)/%.c $(RUNTIME_LIB)
	@mkdir -p $(BIN_DIR)/tests/driver
	@echo "  TEST     driver/$*"
	$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

$(BIN_DIR)/tests/kernel/%: $(TEST_KERNEL_DIR)/%.c $(RUNTIME_LIB)
	@mkdir -p $(BIN_DIR)/tests/kernel
	@echo "  TEST     kernel/$*"
	$(HOST_CC) $(HOST_CFLAGS) $< -L$(BIN_DIR) -lgpgpu $(HOST_LDFLAGS) -o $@

$(BIN_DIR)/tests/cuda/%: $(TEST_CUDA_DIR)/%.c $(RUNTIME_LIB) $(KERNEL_BINS)
	@mkdir -p $(BIN_DIR)/tests/cuda
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

test-driver-%: $(BIN_DIR)/tests/driver/%
	@echo "  RUN      driver/$*"
	@sudo $<

test-kernel-%: $(BIN_DIR)/tests/kernel/%
	@echo "  RUN      kernel/$*"
	@sudo $<

test-cuda-%: $(BIN_DIR)/tests/cuda/%
	@echo "  RUN      cuda/$*"
	@sudo $<

test-python-%: $(PYTHON_EXT)
	@echo "  RUN      python/$*"
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernels && \
		sudo $(PYTHON) $*.py

# -------------------- All Tests -------------------- #

test-driver-all: $(TEST_DRIVER_BINS)
	@echo "=== Running all driver tests ==="
	@for test in $(TEST_DRIVER_BINS); do \
		echo "  RUN      $$(basename $$test)"; \
		sudo $$test; \
	done

test-kernel-all: $(TEST_KERNEL_BINS)
	@echo "=== Running all kernel tests ==="
	@for test in $(TEST_KERNEL_BINS); do \
		echo "  RUN      $$(basename $$test)"; \
		sudo $$test; \
	done

test-cuda-all: $(TEST_CUDA_BINS)
	@echo "=== Running all CUDA tests ==="
	@for test in $(TEST_CUDA_BINS); do \
		echo "  RUN      $$(basename $$test)"; \
		sudo $$test; \
	done

test-python-all: $(PYTHON_EXT)
	@echo "=== Running all Python tests ==="
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernels && \
		for test in $(TEST_PYTHON_NAMES); do \
			echo "  RUN      python/$$test"; \
			sudo $(PYTHON) $$test.py; \
		done

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

.PHONY: run-driver-% run-driver-all
.PHONY: run-kernel-% run-kernel-all
.PHONY: run-cuda-% run-cuda-all
.PHONY: run-python-% run-python-all
.PHONY: build-driver-tests build-kernel-tests build-cuda-tests build-all-tests
.PHONY: clean-tests