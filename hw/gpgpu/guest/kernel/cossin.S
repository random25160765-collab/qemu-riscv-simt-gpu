/*
 * cossin.S — 同时计算 cos 和 sin
 *   out_cos[i] = cos(x[i])
 *   out_sin[i] = sin(x[i])
 *
 * 使用硬件 fcos.s + fsin.s 指令。
 * 常用于旋转位置编码 (RoPE)、傅里叶特征等场景。
 *
 * 输入: N 个 float32 @ 0x100000
 * 输出: cos 结果 @ 0x200000, sin 结果 @ 0x300000
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
    flw  f0, 0(t0)          /* f0 = x */

    /* f1 = cos(x), f0 = sin(x) */
    fsgnj.s f1, f0, f0       /* f1 = x (备份) */
    .word  0x60700053        /* fcos.s f0, f0    f0 = cos(x) */
    fsgnj.s f2, f0, f0       /* f2 = cos(x) */
    fsgnj.s f0, f1, f1       /* f0 = x (恢复) */
    .word  0x60600053        /* fsin.s f0, f0    f0 = sin(x) */

    /* 存储 cos 到 0x200000 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f2, 0(t0)

    /* 存储 sin 到 0x300000 */
    slli t0, t4, 2
    lui  t1, 0x300
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
