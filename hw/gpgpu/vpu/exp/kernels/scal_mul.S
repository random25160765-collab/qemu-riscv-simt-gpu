/*
 * scal_mul.S — VPU 标量乘向量: C[i] = alpha * V[i]
 * alpha @ VRAM[0x400000], V@0x100000, C@0x200000. warp 级 SIMD.
 */

.section .text
.globl _start

_start:
    lui  t0, 0x80000
    lw   t1, 0(t0)

    addi t0, zero, -32
    and  t1, t1, t0

    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t2, 0(t0)
    mul  t3, t3, t2

    bge  t1, t3, exit
    slli t1, t1, 2

    lui  t5, 0x400
    flw  f0, 0(t5)

    addi t0, zero, 4

    lui  t4, 0x100
    add  t4, t4, t1
    vld.v v0, (t4), t0

    vfmul.vs v1, v0, f0

    lui  t4, 0x200
    add  t4, t4, t1
    vst.v v1, (t4), t0

exit:
    ebreak
    .align 2
