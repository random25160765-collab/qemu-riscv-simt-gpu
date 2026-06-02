/*
 * vecmul_vpu.S — VPU 向量加速: C[0..31] = A[0..31] * B[0..31]
 * 单 warp, 32 元素, 测试 vld.v / vfmul.v / vst.v
 */

.section .text
.globl _start

_start:
    addi t0, zero, 4           # stride = 4
    lw   t1, 0x80(zero)             # A base = 0x100000
    vld.v v0, (t1), t0

    lw   t1, 0x84(zero)             # B base = 0x200000
    vld.v v1, (t1), t0

    vfmul.v v2, v0, v1

    lw   t1, 0x88(zero)             # C base = 0x300000
    vst.v v2, (t1), t0

    ebreak
    .align 2
