# GPGPU Guest Top Makefile
# ============================================================
# 三个组件：
#   1. driver/          Linux 内核驱动 gpgpu.ko       → /dev/gpgpu0
#   2. conflux-runtime/ 用户态运行时 libconflux.{a,so}
#   3. pocl/            OpenCL 框架 + Conflux 桥接层
#
# 依赖关系：driver 独立；runtime 依赖 driver/gpgpu_ioctl.h；
#           pocl 依赖 runtime（libconflux.so）和 driver/gpgpu_ioctl.h
#
# 在 QEMU guest 内典型流程：
#   make driver && make install   # 编译并 insmod 内核模块
#   make runtime                  # 编译运行时库
#   make pocl                     # 编译 PoCL（首次较慢）
#   make all                      # 一口气全部
#
# 也可以分别去子目录跑：
#   cd driver && make
#   cd conflux-runtime && make test

DRIVER_DIR  = driver
RUNTIME_DIR = conflux-runtime
POCL_DIR    = pocl
BIN_DIR     = bin

# 设备节点路径（驱动模块名保留 gpgpu）
export GPGPU_DEVICE = /dev/gpgpu0

# 交叉编译工具链前缀（host → RISC-V guest）
RISCV_CROSS ?= riscv64-linux-gnu-

# 内核模块产物（driver/Makefile 里 obj-m := gpgpu.o）
DRIVER_KO   = $(BIN_DIR)/driver/gpgpu.ko

# control.mk 提供 install/remove/dmesg/status 子命令
include scripts/control.mk

# ============================================================
# 1. 内核驱动
# ============================================================

driver: $(DRIVER_KO)

$(DRIVER_KO): $(DRIVER_DIR)/gpgpu.c $(DRIVER_DIR)/gpgpu.h $(DRIVER_DIR)/gpgpu_ioctl.h
	@mkdir -p $(BIN_DIR)/driver
	@echo "  KMOD     gpgpu.ko"
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) modules
	@mv $(DRIVER_DIR)/gpgpu.ko $(BIN_DIR)/driver/
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD)/$(DRIVER_DIR) clean

clean-driver:
	@rm -rf $(BIN_DIR)/driver
	@echo "  CLEAN    driver"

# ============================================================
# 2. Conflux 运行时
# ============================================================

runtime:
	@$(MAKE) -C $(RUNTIME_DIR)

runtime-test:
	@$(MAKE) -C $(RUNTIME_DIR) test

runtime-install:
	@$(MAKE) -C $(RUNTIME_DIR) install

clean-runtime:
	@$(MAKE) -C $(RUNTIME_DIR) clean

# === host 交叉编译给 RISC-V guest ===
# 用法（在 host 上）：
#   make cross-runtime           编译 libconflux 到 build/riscv64-linux-gnu/
#   make cross-runtime-build     编译 lib + 测试可执行文件（不跑测试）
# 然后到 guest：
#   cd conflux-runtime && make test-run    只跑，不重编
cross-runtime:
	@$(MAKE) -C $(RUNTIME_DIR) CROSS_COMPILE=$(RISCV_CROSS)

cross-runtime-build:
	@$(MAKE) -C $(RUNTIME_DIR) test-build CROSS_COMPILE=$(RISCV_CROSS)

# 在 guest 里执行：把 host 交叉编译产物安装到 /usr/local/lib
# 用法（guest 里）：make cross-runtime-install
cross-runtime-install:
	sudo install -m 644 $(RUNTIME_DIR)/build/riscv64-linux-gnu/libconflux.a /usr/local/lib/
	sudo install -m 755 $(RUNTIME_DIR)/build/riscv64-linux-gnu/libconflux.so /usr/local/lib/
	sudo install -d /usr/local/include/conflux
	sudo install -m 644 $(RUNTIME_DIR)/include/*.h /usr/local/include/conflux/
	sudo ldconfig
	@echo "Installed libconflux (riscv64) to /usr/local"

# 在 guest 里执行：把 cross-pocl-stage 产物装到 /usr/local
# 用法（guest 里）：make cross-pocl-install
POCL_INSTALL_STAGE = $(CURDIR)/pocl-install-riscv
cross-pocl-install:
	@if [ ! -d "$(POCL_INSTALL_STAGE)/usr/local" ]; then \
		echo "ERROR: $(POCL_INSTALL_STAGE) 不存在，先在 host 跑 make cross-pocl-stage"; \
		exit 1; \
	fi
	sudo cp -rL $(POCL_INSTALL_STAGE)/usr/local/. /usr/local/
	sudo ldconfig
	@echo "Installed PoCL (riscv64) to /usr/local"

# ============================================================
# 3. PoCL（OpenCL 框架 + Conflux 桥接）
# ============================================================
#
# 两条路径：
#   make pocl         在 guest 内 native 编译（要先 apt install llvm-18-dev 等）
#   make cross-pocl   在 host 上交叉编译（要先做完 stage-riscv-llvm.sh
#                     和 install-riscv-llvm-sysroot.sh，把 RISC-V LLVM 装到 sysroot）

POCL_BUILD       = $(POCL_DIR)/build
POCL_BUILD_CROSS = $(POCL_DIR)/build-cross
RISCV_LLVM_CONFIG  = /usr/local/bin/riscv64-llvm-config-18
# WITH_LLVM_CONFIG 必须是 host 可直接执行的 llvm-config（x86），
# LLVM_DIR 指向 sysroot 里 target 侧的 cmake 配置目录
HOST_LLVM_CONFIG   = /usr/bin/llvm-config-18
RISCV_LLVM_DIR     = /usr/riscv64-linux-gnu/lib/llvm-18/lib/cmake/llvm

# 公共 cmake 选项（开启 conflux 桥接、关闭 host CPU、release 模式）
POCL_CMAKE_OPTS = \
    -DENABLE_GPGPU=ON \
    -DENABLE_ICD=ON \
    -DHAVE_OCL_ICD=1 \
    -DHAVE_OCL_ICD_30_COMPATIBLE=1 \
    -DOCL_ICD_INCLUDE_DIRS=/usr/riscv64-linux-gnu/include \
    -DOCL_ICD_LIBRARIES=/usr/riscv64-linux-gnu/lib/libOpenCL.so \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_LOADABLE_DRIVERS=OFF \
    -DHAVE_DLFCN_H=ON \
    -DENABLE_POCLCC=OFF \
    -DENABLE_SPIRV=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    -DDISABLE_LLVM_LINK_LLVM_DYLIB=ON \
    -DCMAKE_BUILD_TYPE=Release

# --- guest 内 native 编译 ---
pocl: $(POCL_BUILD)/Makefile
	@$(MAKE) -C $(POCL_BUILD)

$(POCL_BUILD)/Makefile:
	@mkdir -p $(POCL_BUILD)
	cd $(POCL_BUILD) && cmake .. $(POCL_CMAKE_OPTS)

pocl-install: pocl
	@$(MAKE) -C $(POCL_BUILD) install

# --- host 上交叉编译（需要先做 sysroot 安装） ---
cross-pocl: $(POCL_BUILD_CROSS)/Makefile
	@$(MAKE) -C $(POCL_BUILD_CROSS)

# 把交叉编译产物暂存到 pocl-install-riscv/，供 guest 里拷贝安装
# 用法（host 上）：make cross-pocl-stage
# 然后在 guest 里：sudo cp -r /mnt/hostshare/pocl-install-riscv/usr/local/. /usr/local/
POCL_INSTALL_STAGE = $(CURDIR)/pocl-install-riscv
cross-pocl-stage: cross-pocl
	@$(MAKE) -C $(POCL_BUILD_CROSS) install DESTDIR=$(POCL_INSTALL_STAGE)
	@echo "staged to $(POCL_INSTALL_STAGE)"
	@echo "on guest: sudo cp -r /mnt/hostshare/pocl-install-riscv/usr/local/. /usr/local/"

$(POCL_BUILD_CROSS)/Makefile:
	@if [ ! -x $(HOST_LLVM_CONFIG) ]; then \
		echo "ERROR: $(HOST_LLVM_CONFIG) 不存在，请安装 llvm-18-dev"; \
		exit 1; \
	fi
	@if [ ! -f $(RISCV_LLVM_DIR)/LLVMConfig.cmake ]; then \
		echo "ERROR: $(RISCV_LLVM_DIR) 不存在"; \
		echo "  先在 guest 跑   sudo bash scripts/stage-riscv-llvm.sh"; \
		echo "  再在 host 上跑  sudo bash scripts/install-riscv-llvm-sysroot.sh"; \
		exit 1; \
	fi
	@mkdir -p $(POCL_BUILD_CROSS)
	cd $(POCL_BUILD_CROSS) && cmake .. \
		-DCMAKE_TOOLCHAIN_FILE=$(CURDIR)/scripts/toolchain-riscv.cmake \
		-DWITH_LLVM_CONFIG=$(HOST_LLVM_CONFIG) \
		-DLLVM_DIR=$(RISCV_LLVM_DIR) \
		$(POCL_CMAKE_OPTS)

clean-pocl:
	@rm -rf $(POCL_BUILD) $(POCL_BUILD_CROSS)
	@echo "  CLEAN    pocl"

# ============================================================
# 开发快捷目标
# ============================================================
#
# Host 端（x86，交叉编译）：
#   make dev-runtime   只重编 runtime（改 conflux-runtime/src/ 后用，快）
#   make dev           重编 runtime + pocl + stage（完整重建，慢）
#
# Guest 端（RISC-V，部署+测试）：
#   make deploy        insmod 驱动 + 安装 runtime + 安装 pocl（首次或全量更新）
#   make deploy-runtime  只重装 runtime（改 runtime 后快速更新）

# Host: 只重编 runtime
dev-runtime:
	@$(MAKE) cross-runtime

# Host: 重编 runtime + pocl + stage
dev:
	@$(MAKE) cross-runtime
	@$(MAKE) cross-pocl
	@$(MAKE) cross-pocl-stage
	@echo "=== cross build + stage done ==="
	@echo "    on guest: make deploy"

# Guest: 全量部署（驱动 + runtime + pocl）
deploy: install cross-runtime-install cross-pocl-install
	@echo "=== deploy done ==="

# Guest: 只更新 runtime（改 runtime 后快速 reload）
deploy-runtime: cross-runtime-install
	@echo "=== runtime deployed ==="

# ============================================================
# 一口气
# ============================================================

all: driver runtime pocl

clean: clean-driver clean-runtime clean-pocl
	@rm -rf $(BIN_DIR)
	@echo "All cleaned."

help:
	@echo "=== Host 端（x86，交叉编译） ==="
	@echo "  make dev-runtime        只重编 runtime（改 conflux-runtime/src/ 后用，快）"
	@echo "  make dev                重编 runtime + pocl + stage（完整重建）"
	@echo "  make cross-runtime      编译 libconflux → build/riscv64-linux-gnu/"
	@echo "  make cross-pocl         交叉编译 PoCL"
	@echo "  make cross-pocl-stage   把 PoCL 安装到 pocl-install-riscv/"
	@echo ""
	@echo "=== Guest 端（RISC-V，部署+测试） ==="
	@echo "  make deploy             全量部署：驱动 + runtime + pocl"
	@echo "  make deploy-runtime     只重装 runtime（改 runtime 后快速更新）"
	@echo "  make install            insmod gpgpu.ko + chmod /dev/gpgpu0"
	@echo "  make remove             rmmod gpgpu"
	@echo "  make status             查看 /dev/gpgpu* 和模块状态"
	@echo "  make cross-runtime-install  安装交叉编译的 libconflux"
	@echo "  make cross-pocl-install     安装交叉编译的 PoCL"
	@echo "  make runtime-test       native 编译 + 跑 SIM 回归测试"
	@echo ""
	@echo "=== 其他 ==="
	@echo "  make driver             编译内核驱动 → bin/driver/gpgpu.ko"
	@echo "  make all                driver + runtime + pocl（guest native）"
	@echo "  make clean              清理全部产物"
	@echo ""
	@echo "Variables:"
	@echo "  RISCV_CROSS             工具链前缀，默认: $(RISCV_CROSS)"

.PHONY: driver clean-driver \
        runtime runtime-test runtime-install clean-runtime \
        cross-runtime cross-runtime-build cross-runtime-install \
        cross-pocl-install \
        pocl pocl-install cross-pocl cross-pocl-stage clean-pocl \
        dev dev-runtime deploy deploy-runtime \
        all clean help
