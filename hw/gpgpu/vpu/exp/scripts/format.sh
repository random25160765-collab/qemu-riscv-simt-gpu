#!/usr/bin/env bash
#
# format.sh — 自动格式化 VPU SIMT engine 代码库
#
# 用法:
#   ./scripts/format.sh             格式化所有 C/H 文件 (原地修改)
#   ./scripts/format.sh --check     CI 模式: 仅检查, 有差异则报错退出
#   ./scripts/format.sh --diff      预览需要修改的 diff, 不修改文件
#   ./scripts/format.sh <file>...   只格式化指定的文件
#
# 要求: 当前目录 (.clang-format 所在目录) 为项目根。

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$PROJ_ROOT/.clang-format" ]; then
    echo "ERROR: .clang-format not found in $PROJ_ROOT" >&2
    exit 1
fi

CLANG_FORMAT="$(which clang-format 2>/dev/null || true)"
if [ -z "$CLANG_FORMAT" ]; then
    echo "ERROR: clang-format not found on PATH" >&2
    echo "Install with: sudo apt install clang-format" >&2
    exit 1
fi

# ---- 解析参数 ----
MODE="format"
TARGET_FILES=()

for arg in "$@"; do
    case "$arg" in
        --check) MODE="check" ;;
        --diff)  MODE="diff"  ;;
        *)       TARGET_FILES+=("$arg") ;;
    esac
done

# ---- 自动收集文件 ----
if [ "${#TARGET_FILES[@]}" -eq 0 ]; then
    cd "$PROJ_ROOT"
    mapfile -t TARGET_FILES < <(
        find . -type f \( -name '*.c' -o -name '*.h' \) \
            ! -path './kernels/*'    \
            ! -name '*.c.inc'        \
        | sort
    )
fi

if [ "${#TARGET_FILES[@]}" -eq 0 ]; then
    echo "No files to format."
    exit 0
fi

echo "clang-format: $($CLANG_FORMAT --version | head -1)"
echo "Root:  $PROJ_ROOT"
echo "Files: ${#TARGET_FILES[@]}"
echo "Mode:  $MODE"
echo "------"

# ---- 执行 ----
NEEDS_FORMAT=()
FORMATTED=0

cd "$PROJ_ROOT"

case "$MODE" in
    check|diff)
        for f in "${TARGET_FILES[@]}"; do
            if [ ! -f "$f" ]; then continue; fi
            if ! diff -q "$f" <($CLANG_FORMAT --style=file "$f") >/dev/null 2>&1; then
                NEEDS_FORMAT+=("$f")
            fi
        done
        if [ "${#NEEDS_FORMAT[@]}" -eq 0 ]; then
            echo "All ${#TARGET_FILES[@]} files are correctly formatted."
            exit 0
        fi
        echo ""
        echo "===== ${#NEEDS_FORMAT[@]} file(s) need formatting ====="
        for f in "${NEEDS_FORMAT[@]}"; do
            echo "  $f"
        done
        if [ "$MODE" == "diff" ]; then
            echo ""
            echo "===== Diff ====="
            for f in "${NEEDS_FORMAT[@]}"; do
                echo ""
                echo "--- $f"
                diff -u "$f" <($CLANG_FORMAT --style=file "$f") || true
            done
        fi
        echo ""
        echo "Run 'make format' to fix."
        exit 1
        ;;
    format)
        for f in "${TARGET_FILES[@]}"; do
            if [ ! -f "$f" ]; then continue; fi
            echo "  formatting: $f"
            $CLANG_FORMAT -i --style=file "$f"
            FORMATTED=$((FORMATTED + 1))
        done
        echo ""
        echo "Done: formatted $FORMATTED file(s)."
        ;;
esac
