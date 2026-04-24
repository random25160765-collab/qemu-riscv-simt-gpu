# toolchain.mk - RISC-V Python 工具链安装管理
# ============================================================

PROJECT_ROOT := $(shell pwd)
TOOLCHAIN_DIR = $(PROJECT_ROOT)/toolchain
TOOLCHAIN_TAR = toolchain-riscv-python.tar.gz
TOOLCHAIN_TAR_PATH = $(TOOLCHAIN_DIR)/$(TOOLCHAIN_TAR)

TOOLCHAIN_EXISTS = $(shell [ -f "$(TOOLCHAIN_DIR)/include/python3.13/Python.h" ] && echo "yes")

setup-toolchain:
	@if [ "$(TOOLCHAIN_EXISTS)" = "yes" ]; then \
		echo "Toolchain already installed at $(TOOLCHAIN_DIR)"; \
		exit 0; \
	fi
	@if [ ! -f "$(TOOLCHAIN_TAR_PATH)" ]; then \
		echo "Error: $(TOOLCHAIN_TAR) not found"; \
		echo "Please place $(TOOLCHAIN_TAR) in $(TOOLCHAIN_DIR)"; \
		exit 1; \
	fi
	@echo "Extracting RISC-V Python toolchain..."
	@cd $(TOOLCHAIN_DIR) && tar -xzf $(TOOLCHAIN_TAR)
	@echo "Toolchain installed at $(TOOLCHAIN_DIR)"

clean-toolchain:
#   WARNING: Don't modify this line!!!! 
	@rm -rf $(TOOLCHAIN_DIR)/include $(TOOLCHAIN_DIR)/lib
	@echo "Toolchain cleaned (preserving $(TOOLCHAIN_TAR))"

check-toolchain:
	@if [ "$(TOOLCHAIN_EXISTS)" = "yes" ]; then \
		echo "✓ Toolchain found at $(TOOLCHAIN_DIR)"; \
	else \
		echo "✗ Toolchain not found"; \
		echo "Run: make setup-toolchain"; \
		exit 1; \
	fi

RISCV_PYTHON_INC = $(TOOLCHAIN_DIR)/include/python3.13
RISCV_PYTHON_LIB = $(TOOLCHAIN_DIR)/lib

.PHONY: setup-toolchain clean-toolchain check-toolchain