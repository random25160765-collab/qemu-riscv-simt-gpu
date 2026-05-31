/*
 * gelu.S — 元素级 GELU 激活函数
 *   f(x) = x * sigmoid(1.702 * x)
 *
 * GELU 近似公式（GPT/BERT 等 Transformer 的标准实现）。
 * 注意：x * sigmoid(x) 是 SiLU/Swish，不是 GELU。
 *
 * 输入: N 个 float32 @ 0x100000
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
    flw  f0, 0(t0)          /* f0 = x */

    /* gelu(x) = x * sigmoid(1.702 * x) */
    fsgnj.s f1, f0, f0      /* f1 = x (备份) */
    li   t1, 0x3FD9D1A3     /* 1.702 的 IEEE 754 编码 */
    fmv.w.x f2, t1           /* f2 = 1.702 */
    fmul.s f0, f0, f2        /* f0 = 1.702 * x */
    .word 0x60500053        /* fsigmoid.s f0, f0 */
    fmul.s f0, f0, f1        /* f0 = x * sigmoid(1.702*x) */

    /* 存储结果 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
