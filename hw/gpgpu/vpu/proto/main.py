#!/usr/bin/env python3
"""
协议宏生成入口
用法：python3 main.py [输出目录]
  生成 pt_inst.h (level=0x00) 和 pt_event.h (level=0x01)
"""
import sys
import os
import gen_inst
import gen_event

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

gen_inst.generate(os.path.join(OUT_DIR, "pt_inst.h"))
gen_event.generate(os.path.join(OUT_DIR, "pt_event.h"))
