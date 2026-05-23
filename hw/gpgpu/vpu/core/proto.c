/*
 * 二进制指令 Trace 实现
 * 数据直接拼接写入 inst_trace.bin
 */
#include "qemu/osdep.h"
#include "proto.h"

static FILE *inst_trace_fp;

static void inst_trace_reopen(void)
{
    if (inst_trace_fp) {
        fclose(inst_trace_fp);
    }
    inst_trace_fp = fopen("inst_trace.bin", "ab");
    if (inst_trace_fp) {
        setvbuf(inst_trace_fp, NULL, _IONBF, 0);
    }
}

static bool inst_trace_write(const void *buf, size_t size)
{
    if (!inst_trace_fp) {
        inst_trace_reopen();
    }
    if (!inst_trace_fp) {
        return false;
    }
    if (fwrite(buf, size, 1, inst_trace_fp) != 1) {
        /* fd 可能被 QEMU 关闭，尝试重新打开并写入 */
        inst_trace_reopen();
        if (!inst_trace_fp) {
            return false;
        }
        return fwrite(buf, size, 1, inst_trace_fp) == 1;
    }
    return true;
}

void gpgpu_inst_trace_bin(uint32_t inst_code, ...)
{
    uint32_t nargs = (inst_code >> 24) & 0xF;

    if (!inst_trace_write(&inst_code, 4)) {
        return;
    }

    va_list args;
    va_start(args, inst_code);
    for (uint32_t i = 0; i < nargs; i++) {
        uint32_t val = va_arg(args, uint32_t);
        if (!inst_trace_write(&val, 4)) {
            va_end(args);
            return;
        }
    }
    va_end(args);
}
