/*
 * rsqrt.S — 元素级倒数平方根
 *   f(x) = 1 / sqrt(x)
 *
 * 使用硬件 frsqrt.s 指令，单条指令完成。
 * 广泛用于 BatchNorm/LayerNorm/RMSNorm 等归一化操作。
 *
 * 输入: N 个 float32 @ 0x100000 (必须 > 0)
 * 输出: N 个 float32 @ 0x200000
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

    /* 边界检查 */
    bge  t4, t3, exit

    /* 加载输入 */
    slli t0, t4, 2
    lui  t1, 0x100
    add  t0, t1, t0
    flw  f0, 0(t0)

    /* f0 = 1/sqrt(f0) */
    .word 0x60300053        /* frsqrt.s f0, f0 */

    /* 存储结果 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
