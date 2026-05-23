/*
 * 二进制指令 Trace 协议
 * 格式：inst_code(4B) + nargs×operand(4B)，直接拼接写入文件
 * nargs 已编码在 inst_code bits[27:24]
 */
#ifndef GPGPU_PROTO_H
#define GPGPU_PROTO_H

#include <stdint.h>
#include "proto/pt_inst.h"

void gpgpu_inst_trace_bin(uint32_t inst_code, ...);

#define GPGPU_INST_BIN(inst_code, ...) \
    gpgpu_inst_trace_bin(inst_code, ##__VA_ARGS__)

#endif /* GPGPU_PROTO_H */
