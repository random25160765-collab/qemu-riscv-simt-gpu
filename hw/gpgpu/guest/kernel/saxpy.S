/*
 * saxpy.S — 融合标量乘加 (Scalar Alpha X Plus Y)
 *   C[i] = alpha * A[i] + B[i]
 *
 * 这是 BLAS Level 1 最常用的操作。单趟融合避免两次 kernel 调用。
 *
 * 输入: A[N] @ 0x100000, B[N] @ 0x200000
 * 标量: alpha @ 0x400000 (float32)
 * 输出: C[N] @ 0x300000
 * 参数: grid_dim[0] = N
 */

.section .text
.globl _start

_start:
    /* threadIdx.x */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* blockIdx.x */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)

    /* N = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)

    /* global_id = blockIdx.x * 256 + threadIdx.x */
    addi t4, zero, 256
    mul  t2, t2, t4
    add  t4, t2, t1

    /* 加载 alpha 标量 */
    lui  t0, 0x400
    flw  f2, 0(t0)          /* f2 = alpha */

    /* 边界检查 */
    bge  t4, t3, exit

    /* 加载 A[i] */
    slli t0, t4, 2
    lui  t1, 0x100
    add  t0, t1, t0
    flw  f0, 0(t0)

    /* alpha * A[i] */
    fmul.s f0, f0, f2

    /* 加载 B[i] */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    flw  f1, 0(t0)

    /* C[i] = alpha * A[i] + B[i] */
    fadd.s f0, f0, f1

    /* 存储到 0x300000 */
    slli t0, t4, 2
    lui  t1, 0x300
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
