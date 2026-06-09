# inst/Makefile — ISA 扩展 handler 生成
#
# 用法:
#   make -C inst           生成所有 handler + dispatch 表
#   make -C inst clean     清除所有生成产物
#
# 主 Makefile 在构建前自动调用本目标。

.PHONY: all clean handlers dispatch

all: handlers dispatch

handlers:
	@python3 build_all.py

dispatch:
	@python3 gen_dispatch.py

clean:
	@python3 build_all.py --clean
