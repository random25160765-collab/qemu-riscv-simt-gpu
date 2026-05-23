/*
 * softmax_norm.S — Softmax 归一化 Pass
 *
 * 完整 softmax 流程（两趟）：
 *   1. softmax_exp  kernel: temp[i] = e^{x[i]}    → @0x200000
 *   2. CPU: sum = Σ temp[i];  写入 sum            → @0x80000040
 *   3. softmax_norm kernel: out[i] = temp[i] / sum → @0x300000
 *
 * Step 3 使用硬件 frcp.s 指令在 GPU 侧计算 1/sum。
 *
 * 参数:
 *   grid_dim[0] = N
 *   sum @ 0x400000 (由 host 在两次 kernel 之间写入)
 *
 * 输入:  temp (exp 值) @ 0x200000
 * 输出:  softmax 结果   @ 0x300000
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

    /* 读 sum = Σ e^{x}，用 frcp.s 算 1/sum */
    lui  t0, 0x400
    flw  f1, 0(t0)          /* f1 = sum */
    .word 0x60200053        /* frcp.s f1, f1    f1 = 1/sum */

    /* 边界检查 */
    bge  t4, t3, exit

    /* 读 temp[i] = e^{x[i]} */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    flw  f0, 0(t0)

    /* out[i] = e^{x[i]} / sum */
    fmul.s f0, f0, f1

    /* 存储到 0x300000 */
    slli t0, t4, 2
    lui  t1, 0x300
    add  t0, t1, t0
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
