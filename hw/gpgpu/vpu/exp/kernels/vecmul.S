/*
 * vecmul.S — RVV 向量乘法: C[i]=A[i]*B[i]
 * warp 级 SIMD, RVV 1.0 标准语法
 */

.section .text
.globl _start

_start:
    /* threadIdx.x (per-lane) */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* warp_base = threadIdx.x & ~31 */
    addi t0, zero, -32
    and  t1, t1, t0

    /* global_index = blockIdx.x * blockDim.x + warp_base */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t6, 0(t0)              /* blockIdx.x */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t2, 0(t0)              /* blockDim.x */
    mul  t6, t6, t2             /* blockIdx.x * blockDim.x */
    add  t1, t1, t6             /* global_offset = warp_base + blockIdx.x*blockDim.x */

    /* N = gridDim.x * blockDim.x (reuse t2 = blockDim.x) */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)
    mul  t3, t3, t2

    /* 边界: global_offset >= N → exit */
    bge  t1, t3, exit

    /* byte offset = global_offset * 4 */
    slli t1, t1, 2

    lw   t4, 0x80(zero)
    add  t4, t4, t1
    vle32.v v0, (t4)

    lw   t4, 0x84(zero)
    add  t4, t4, t1
    vle32.v v1, (t4)

    vfmul.vv v2, v0, v1

    lw   t4, 0x88(zero)
    add  t4, t4, t1
    vse32.v v2, (t4)

exit:
    ebreak
    .align 2
