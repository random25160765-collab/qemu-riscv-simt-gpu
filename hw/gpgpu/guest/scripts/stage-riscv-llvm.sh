#!/bin/bash
# stage-riscv-llvm.sh —— 在 RISC-V guest 内运行
#
# 作用：装 PoCL 编译需要的 dev 包，把 RISC-V 版的 LLVM/clang/hwloc 等导出到
#       /mnt/hostshare/_riscv-llvm-stage/，供 host 端的安装脚本搬到 sysroot。
#
# 用法（guest 内）：
#   sudo bash scripts/stage-riscv-llvm.sh

set -e

STAGE=/mnt/hostshare/_riscv-llvm-stage
LLVM_VER=18

if [ "$EUID" -ne 0 ]; then
    echo "请用 sudo 运行：sudo bash $0" >&2
    exit 1
fi

if [ ! -d /mnt/hostshare ]; then
    echo "/mnt/hostshare 不存在，无法 stage（需要 host/guest 共享目录）" >&2
    exit 1
fi

echo "[1/4] 安装 PoCL 编译依赖..."
apt-get update
apt-get install -y \
    llvm-${LLVM_VER}-dev \
    libclang-${LLVM_VER}-dev \
    clang-${LLVM_VER} \
    libpolly-${LLVM_VER}-dev \
    libhwloc-dev \
    ocl-icd-opencl-dev \
    pkg-config

echo "[2/4] 准备 stage 目录 ${STAGE}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/lib" "${STAGE}/include" "${STAGE}/bin"

# ---- 关键约定（千万别改！）----
# Debian/Ubuntu 在 /usr/lib/llvm-18/include/ 下放的 llvm/llvm-c/clang 都是
# 符号链接，指向 /usr/include/llvm-18/、/usr/include/llvm-c/ 等。
# 我们必须用 `cp -rL`（递归 + 跟随符号链接）把它们展开成真实文件，
# 否则 stage 出来的是"指向 stage 之外不存在路径的死链接"，
# 在 host 上装到 sysroot 后会变成 ASCII 文本占位符，编译时 #include 找不到。
# 所有 include 目录都必须用 -rL，不能用 -a / -r。

echo "[3/4] 拷贝 LLVM 主目录..."
# llvm-18 主目录里大部分是真实文件（cmake 配置、bin 等），用 -a 即可
cp -a /usr/lib/llvm-${LLVM_VER} "${STAGE}/lib/"

# 但 include 子目录里有符号链接，必须重做：先删掉刚 cp 进来的那份，再用 -rL 展开
rm -rf "${STAGE}/lib/llvm-${LLVM_VER}/include"
mkdir -p "${STAGE}/lib/llvm-${LLVM_VER}/include"
cp -rL /usr/lib/llvm-${LLVM_VER}/include/. "${STAGE}/lib/llvm-${LLVM_VER}/include/"

# 验证：StringRef.h 必须是真实文件，不能是 ASCII 占位
if ! [ -f "${STAGE}/lib/llvm-${LLVM_VER}/include/llvm/ADT/StringRef.h" ] || \
   file "${STAGE}/lib/llvm-${LLVM_VER}/include/llvm/ADT/StringRef.h" | grep -q "ASCII text"; then
    echo "ERROR: stage 里 StringRef.h 不是真实文件！cp -rL 失败了" >&2
    exit 1
fi
echo "  ✓ LLVM 头文件展开成功（StringRef.h 是真实文件）"

echo "[4/4] 拷贝架构库 + 其他依赖..."
# LLVM 运行时直接依赖（libLLVM-18.so 需要这些库）
# 用 -L 展开符号链接确保拷出真实 ELF 文件
RISCV_LIBDIR=/usr/lib/riscv64-linux-gnu
for lib in \
    libzstd.so.1 \
    libxml2.so.2 \
    libz.so.1 \
    libffi.so.8 \
    libedit.so.2 \
    libz3.so.4 \
    libLLVM*.so* \
    libclang*.so* \
    libPolly*.so* \
    libhwloc*.so* \
    libOpenCL*.so*; do
    for f in $RISCV_LIBDIR/$lib; do
        [ -e "$f" ] && cp -Lf "$f" "${STAGE}/lib/" 2>/dev/null || true
    done
done

# pkgconfig（PoCL 用 pkg-config 找 hwloc）
[ -d /usr/lib/riscv64-linux-gnu/pkgconfig ] && \
    cp -a /usr/lib/riscv64-linux-gnu/pkgconfig "${STAGE}/lib/" || true

# 独立头文件（hwloc / OpenCL CL header）—— 也要 -rL 防止潜在的符号链接
for d in hwloc hwloc.h CL; do
    [ -e "/usr/include/$d" ] && cp -rL "/usr/include/$d" "${STAGE}/include/" 2>/dev/null || true
done

# llvm-config 二进制（host 跑不了 RISC-V 二进制，留作参考）
[ -f /usr/bin/llvm-config-${LLVM_VER} ] && \
    cp /usr/bin/llvm-config-${LLVM_VER} "${STAGE}/bin/" || true

# 修正所有权（hostshare 文件系统可能不支持 chown，忽略错误）
chown -R "${SUDO_USER}:${SUDO_USER}" "${STAGE}" 2>/dev/null || true

cat <<EOF

✓ Stage 完成：${STAGE}
  - 大小：$(du -sh "${STAGE}" | awk '{print $1}')
  - 文件数：$(find "${STAGE}" | wc -l)
  - StringRef.h 类型：$(file "${STAGE}/lib/llvm-${LLVM_VER}/include/llvm/ADT/StringRef.h" | cut -d: -f2-)

下一步（在 host 上）：
  cd <repo>
  sudo bash hw/gpgpu/guest/scripts/install-riscv-llvm-sysroot.sh
EOF
