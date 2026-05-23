/*
 * elu.S — 元素级 ELU 激活函数 (无分支实现)
 *   f(x) = x              if x >= 0
 *        = alpha*(e^x - 1) if x < 0
 *
 * 分支形式在 SIMT 下会导致 divergence 错误，这里用 min/max 实现：
 *   elu(x) = max(0, x) + min(0, alpha * (e^x - 1))
 *
 * 使用硬件 fexp.s 指令，alpha 默认 = 1.0。
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

    /* ── 无分支 ELU ── */

    /* 备份 x */
    fsgnj.s f1, f0, f0      /* f1 = x */

    /* 计算 alpha * (e^x - 1) */
    .word  0x60000053        /* fexp.s f0, f0    f0 = e^x */
    addi   t1, zero, 1
    fcvt.s.w f2, t1          /* f2 = 1.0 */
    fsub.s f0, f0, f2        /* f0 = e^x - 1 */
    /* alpha=1.0 时跳过乘法 */

    /* elu = max(0, x) + min(0, alpha*(e^x - 1)) */
    fcvt.s.w f2, zero        /* f2 = 0.0 */
    fmax.s f3, f1, f2        /* f3 = max(0, x) */
    fmin.s f0, f0, f2        /* f0 = min(0, e^x - 1) */
    fadd.s f0, f3, f0        /* f0 = max(0,x) + min(0,e^x-1) */

    /* 存储结果 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
