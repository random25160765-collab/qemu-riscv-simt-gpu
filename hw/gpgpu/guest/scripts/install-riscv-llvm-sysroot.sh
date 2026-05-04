#!/bin/bash
# install-riscv-llvm-sysroot.sh —— 在 host 上运行
#
# 作用：把 stage-riscv-llvm.sh 导出的 RISC-V LLVM 文件搬到 host 的
#       RISC-V sysroot：/usr/riscv64-linux-gnu/，并装一个 host-runnable
#       的 llvm-config wrapper，供 PoCL cmake 用。
#
# 用法：sudo bash hw/gpgpu/guest/scripts/install-riscv-llvm-sysroot.sh

set -e

# stage 目录：脚本在 scripts/，guest 目录是 scripts/ 的上一级
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUEST_DIR="$(dirname "${SCRIPT_DIR}")"
STAGE="${GUEST_DIR}/_riscv-llvm-stage"
SYSROOT=/usr/riscv64-linux-gnu
LLVM_VER=18
WRAPPER=/usr/local/bin/riscv64-llvm-config-${LLVM_VER}

if [ "$EUID" -ne 0 ]; then
    echo "请用 sudo 运行：sudo bash $0" >&2
    exit 1
fi

if [ ! -d "${STAGE}" ]; then
    echo "未找到 ${STAGE}，先在 guest 上跑 scripts/stage-riscv-llvm.sh" >&2
    exit 1
fi

# ---- 前置校验：stage 必须包含真实头文件，不能有占位符 ----
SR="${STAGE}/lib/llvm-${LLVM_VER}/include/llvm/ADT/StringRef.h"
if [ ! -f "${SR}" ] || file "${SR}" | grep -q "ASCII text"; then
    echo "ERROR: stage 里 StringRef.h 不是真实头文件" >&2
    echo "  请在 guest 上重跑 stage-riscv-llvm.sh（已修复 cp -rL 展开符号链接）" >&2
    exit 1
fi

echo "[1/3] 把 stage 文件安装到 ${SYSROOT}..."

# LLVM 主目录：先彻底清掉旧状态（防御已有占位符），再拷
rm -rf "${SYSROOT}/lib/llvm-${LLVM_VER}"
cp -a "${STAGE}/lib/llvm-${LLVM_VER}" "${SYSROOT}/lib/"

# llvm-18/lib 里的 .so 文件可能是 linker script（Debian 用符号链接），
# 用 -L 展开覆盖确保是真实 ELF
for f in "${STAGE}/lib/llvm-${LLVM_VER}/lib/"lib*.so*; do
    [ -f "$f" ] || continue
    if file "$f" | grep -q "ASCII text"; then
        # 是 linker script / 符号链接占位符，用 -L 拷真实文件
        cp -Lf "$f" "${SYSROOT}/lib/llvm-${LLVM_VER}/lib/" 2>/dev/null || true
    fi
done

# 架构库（lib/*.so*）—— 用 -L 展开所有符号链接，确保是真实 ELF
for f in "${STAGE}/lib/"lib*.so*; do
    [ -f "$f" ] || continue
    cp -Lf "$f" "${SYSROOT}/lib/" 2>/dev/null || \
    cp -f  "$f" "${SYSROOT}/lib/" 2>/dev/null || true
done
# 确保所有库文件可读（cp -L 可能保留了 guest 上只有 root 可读的权限）
chmod a+r "${SYSROOT}/lib/"lib*.so* 2>/dev/null || true

# pkgconfig
if [ -d "${STAGE}/lib/pkgconfig" ]; then
    mkdir -p "${SYSROOT}/lib/pkgconfig"
    cp -a "${STAGE}/lib/pkgconfig/." "${SYSROOT}/lib/pkgconfig/" 2>/dev/null || true
fi

# stage/include/ 里的独立头文件（hwloc、CL 等）
[ -d "${STAGE}/include" ] && \
    cp -a "${STAGE}/include/." "${SYSROOT}/include/" 2>/dev/null || true

# ---- 后置校验：装到 sysroot 后再确认一遍 ----
SYSR_SR="${SYSROOT}/lib/llvm-${LLVM_VER}/include/llvm/ADT/StringRef.h"
if [ ! -f "${SYSR_SR}" ] || file "${SYSR_SR}" | grep -q "ASCII text"; then
    echo "ERROR: 装到 sysroot 后 StringRef.h 仍是占位符。这不该发生。" >&2
    file "${SYSR_SR}" >&2
    exit 1
fi
echo "  ✓ sysroot 里 StringRef.h 是真实头文件"

echo "[2/3] 安装 llvm-config wrapper + clang 软链接..."

# wrapper：输出 --libdir / --includedir 指向 RISC-V sysroot；
# 输出 --bindir 仍用 host 路径，让 PoCL 找到 host 可执行的 clang
cat > "${WRAPPER}" <<'WEOF'
#!/bin/bash
LLVM_VER=18
SYSROOT=/usr/riscv64-linux-gnu
HOST_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VER}

[ -x "${HOST_LLVM_CONFIG}" ] || { echo "host llvm-config-${LLVM_VER} not found" >&2; exit 1; }

OUTPUT="$("${HOST_LLVM_CONFIG}" "$@")"

# 对 --bindir / --prefix 不替换路径（保持 host 路径，让 PoCL 找到 host clang）
if [[ "$*" == *"--bindir"* ]] || [[ "$*" == *"--prefix"* ]]; then
    echo "${OUTPUT}"
else
    echo "${OUTPUT}" | sed \
        -e "s|/usr/lib/llvm-${LLVM_VER}|${SYSROOT}/lib/llvm-${LLVM_VER}|g" \
        -e "s|/usr/lib/x86_64-linux-gnu|${SYSROOT}/lib|g" \
        -e "s|/usr/include|${SYSROOT}/include|g"
fi
WEOF
chmod +x "${WRAPPER}"
ln -sf "${WRAPPER}" "/usr/local/bin/riscv64-llvm-config"

# PoCL 在 llvm-config --bindir 返回的目录里找 clang-18；
# 该目录是 /usr/lib/llvm-18/bin，确保 host 的 clang-18 在那里
if [ ! -f /usr/lib/llvm-18/bin/clang-18 ] && [ -x /usr/bin/clang-18 ]; then
    mkdir -p /usr/lib/llvm-18/bin
    ln -sf /usr/bin/clang-18 /usr/lib/llvm-18/bin/clang-18
    echo "  linked /usr/lib/llvm-18/bin/clang-18 -> /usr/bin/clang-18"
fi

echo "[3/3] 验证..."
echo "  ${WRAPPER} --version    : $(${WRAPPER} --version 2>&1)"
echo "  ${WRAPPER} --libdir     : $(${WRAPPER} --libdir 2>&1)"
echo "  ${WRAPPER} --includedir : $(${WRAPPER} --includedir 2>&1)"
echo "  StringRef.h type        : $(file "${SYSR_SR}" | cut -d: -f2-)"

cat <<EOF

✓ 安装完成。

下一步（在 host 上交叉编译 PoCL）：
  cd hw/gpgpu/guest
  make cross-pocl
EOF
