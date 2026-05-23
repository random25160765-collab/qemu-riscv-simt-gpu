/*
 * softmax.S — 完整 Softmax（自包含，单 kernel）
 *   softmax(x_i) = e^{x_i} / Σ_j e^{x_j}
 *
 * 每个线程独立遍历全部 N 个输入计算 sum，避免跨线程 reduction。
 * 总计算量 O(N²)，对模拟器可接受。
 *
 * 输入:  x[N] @ 0x100000
 * 输出:  y[N] @ 0x200000
 * 参数:  grid_dim[0] = N
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
    lw   t3, 0(t0)          /* t3 = N */

    /* global_id = blockIdx.x * 256 + threadIdx.x */
    addi t4, zero, 256
    mul  t2, t2, t4
    add  t4, t2, t1         /* t4 = global_id */

    /* 边界检查 */
    bge  t4, t3, exit

    /* 加载 x[global_id]，计算 my_exp = e^{x_i} */
    slli t0, t4, 2
    lui  t1, 0x100
    add  t0, t1, t0
    flw  f0, 0(t0)          /* f0 = x_i */
    .word 0x60000053        /* fexp.s f0, f0    f0 = my_exp */

    /* 初始化 sum = 0.0 */
    fcvt.s.w f1, zero       /* f1 = sum */

    /* j = 0 */
    addi t0, zero, 0        /* t0 = j */

loop:
    bge  t0, t3, done       /* j >= N */

    /* 加载 x[j] */
    slli t1, t0, 2
    lui  t2, 0x100
    add  t1, t2, t1
    flw  f2, 0(t1)          /* f2 = x[j] */

    /* sum += e^{x[j]} */
    .word 0x60010053        /* fexp.s f2, f2 */
    fadd.s f1, f1, f2

    /* j++ */
    addi t0, t0, 1
    j    loop

done:
    /* softmax = my_exp / sum */
    fdiv.s f0, f0, f1

    /* 存储到 0x200000 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
