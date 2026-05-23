# GPGPU Project Test Makefile
# ============================================================

# ============================ Run Tests ================================

PYTHON = python3

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

test-python-%: $(RISCV_PY_EXT)
	@echo "  RUN      python/$*"
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernel && \
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

test-python-all: $(RISCV_PY_EXT)
	@echo "=== Running all Python tests ==="
	@cd $(TEST_PYTHON_DIR) && \
		export GPGPU_KERNEL_DIR=$(PWD)/$(BIN_DIR)/kernel && \
		for test in $(TEST_PYTHON_NAMES); do \
			echo "  RUN      python/$$test"; \
			sudo $(PYTHON) $$test.py; \
		done

.PHONY: test-driver-% test-driver-all
.PHONY: test-kernel-% test-kernel-all
.PHONY: test-cuda-% test-cuda-all
.PHONY: test-python-% test-python-all