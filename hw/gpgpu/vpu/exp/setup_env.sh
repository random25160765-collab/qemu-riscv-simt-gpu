#!/usr/bin/env bash
# ===========================================================================
# setup_env.sh — 安装 VPU Simulator C 部分所需的系统依赖
#
# 用法:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
#   sudo 权限用于 apt-get install，脚本内会自动调用 sudo。
#   幂等：重复运行不会重复安装已存在的包。
# ===========================================================================

set -euo pipefail

# ── 颜色 ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; }
info() { echo -e "       $*"; }

# ── 检测函数 ──────────────────────────────────────────────────────

has_cmd()       { command -v "$1" &>/dev/null; }
has_pkg()       { dpkg -s "$1" &>/dev/null 2>&1; }
has_header()    { local hdr="$1"; shift; printf '#include <%s>\n' "$hdr" | gcc -fsyntax-only -x c - "$@" &>/dev/null 2>&1; }
has_header_at() { [ -f "$1" ]; }
has_lib()       { ldconfig -p 2>/dev/null | grep -q "$1"; }
has_py_module() { python3 -c "import $1" &>/dev/null 2>&1; }

need_pkg() {
    # 检查包是否已安装，未安装则用 apt 安装
    if has_pkg "$1"; then
        ok "$1"
    else
        warn "missing $1 — installing..."
        sudo apt-get install -y "$1"
        ok "$1 (installed)"
    fi
}

# ── 标题 ──────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo " VPU Simulator — 系统环境配置 (C 部分)"
echo "=============================================="
echo ""

# ── 1. 更新 apt 缓存 ─────────────────────────────────────────────

info "更新 apt 包缓存..."
sudo apt-get update -qq
ok "apt cache up-to-date"

echo ""

# ── 2. 主机 C 工具链 ─────────────────────────────────────────────

echo "── 主机 C 工具链 ──────────────────────────────────"
need_pkg "build-essential"

# 验证关键组件
for cmd in gcc make; do
    if has_cmd "$cmd"; then
        ok "$cmd 可用 ($(which $cmd))"
    else
        fail "$cmd 未找到!"
        exit 1
    fi
done

echo ""

# ── 3. 标准头文件 / 库 ──────────────────────────────────────────

echo "── 标准 C 头文件 & 运行时库 ────────────────────────"
for hdr in stdio.h stdlib.h string.h math.h pthread.h fenv.h unistd.h \
           sys/stat.h fcntl.h time.h stdint.h; do
    if has_header "$hdr"; then
        ok "<$hdr>"
    else
        fail "<$hdr> 不可用 (检查 build-essential/libc6-dev)"
        exit 1
    fi
done

echo ""

# ── 4. Lua 5.4 开发库 ───────────────────────────────────────────

echo "── Lua 5.4 开发库 ──────────────────────────────────"
need_pkg "liblua5.4-dev"
need_pkg "liblua5.4-0"

# 验证头文件和库
# Lua 5.4 头文件安装在 /usr/include/lua5.4/，需要额外 -I 路径
LUA_INC="-I/usr/include/lua5.4"
for hdr in lua.h lualib.h lauxlib.h; do
    if has_header "$hdr" $LUA_INC; then
        ok "<$hdr>"
    else
        fail "<$hdr> 不可用"
        exit 1
    fi
done

if has_lib "liblua5.4"; then
    ok "liblua5.4.so"
else
    fail "liblua5.4.so 未找到"
    exit 1
fi

echo ""

# ── 5. RISC-V 交叉编译器 (bare-metal) ───────────────────────────

echo "── RISC-V 交叉编译器 (bare-metal) ──────────────────"
need_pkg "gcc-riscv64-unknown-elf"
need_pkg "binutils-riscv64-unknown-elf"

# 验证
if has_cmd "riscv64-unknown-elf-gcc"; then
    ok "riscv64-unknown-elf-gcc ($(which riscv64-unknown-elf-gcc))"
else
    fail "riscv64-unknown-elf-gcc 未找到!"
    exit 1
fi

if has_cmd "riscv64-unknown-elf-objcopy"; then
    ok "riscv64-unknown-elf-objcopy ($(which riscv64-unknown-elf-objcopy))"
else
    fail "riscv64-unknown-elf-objcopy 未找到!"
    exit 1
fi

echo ""

# ── 6. Python 及依赖 ────────────────────────────────────────────

echo "── Python 3 & YAML ─────────────────────────────────"
if has_cmd "python3"; then
    ok "python3 ($(python3 --version 2>&1))"
else
    fail "python3 未安装"
    exit 1
fi

need_pkg "python3-yaml"

if has_py_module "yaml"; then
    ok "python3 yaml 模块可用"
else
    fail "python3 yaml 模块不可用"
    exit 1
fi

echo ""

# ── 完成 ──────────────────────────────────────────────────────────

echo "=============================================="
echo -e "${GREEN}  所有依赖已就绪，可以运行 make test${NC}"
echo "=============================================="
echo ""
echo "  make test            # 编译并运行所有测试"
echo "  make clean           # 清理构建产物"
echo "  make count           # 统计代码行数"
echo "  make help            # 查看所有目标"
echo ""
