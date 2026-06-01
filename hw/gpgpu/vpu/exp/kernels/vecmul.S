/*
 * vecmul.S — 元素级向量乘法 (标准 CUDA grid/block 模式)
 *   C[i] = A[i] * B[i]
 *
 * global_id = blockIdx.x * blockDim.x + threadIdx.x
 * total     = gridDim.x  * blockDim.x
 *
 * 输入: A[N] @ 0x100000, B[N] @ 0x200000
 * 输出: C[N] @ 0x300000
 */

.section .text
.globl _start

_start:
    /* blockDim.x */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t5, 0(t0)

    /* threadIdx.x */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* blockIdx.x */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)

    /* global_id = blockIdx.x * blockDim.x + threadIdx.x */
    mul  t2, t2, t5
    add  t4, t2, t1

    /* N = gridDim.x * blockDim.x */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)
    mul  t3, t3, t5

    /* 边界检查 */
    bge  t4, t3, exit

    /* 加载 A[i] */
    slli t0, t4, 2
    lui  t1, 0x100
    add  t0, t1, t0
    flw  f0, 0(t0)

    /* 加载 B[i] */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    flw  f1, 0(t0)

    /* C[i] = A[i] * B[i] */
    fmul.s f0, f0, f1

    /* 存储到 0x300000 */
    slli t0, t4, 2
    lui  t1, 0x300
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
