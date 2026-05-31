/*
 * 矩阵乘法 GPU Kernel
 * C = A * B  (M x K) * (K x N) = (M x N)
 * 
 * 参数通过 GPU 核心内部寄存器传递：
 * - grid_dim[0] = M (0x80000030)
 * - grid_dim[1] = K (0x80000034)
 * - block_dim[0] = N (0x80000020)
 */

.section .text
.globl _start

_start:
    /* 读取 thread_id.x = col */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* 读取 block_id.x = row */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)

    /* 读取 M = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)

    /* 读取 N = block_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t4, 0(t0)

    /* 读取 K = grid_dim[1] */
    lui  t0, 0x80000
    addi t0, t0, 0x34
    lw   t5, 0(t0)

    /* 边界检查 */
    bge  t2, t3, exit
    bge  t1, t4, exit

    /* sum = 0.0 */
    fcvt.s.w f0, zero

    /* k = 0 — 在循环外面！ */
    addi t6, zero, 0

loop:
    /* if k >= K: goto done */
    bge  t6, t5, done

    /* A地址 */
    mul  a0, t2, t5
    add  a0, a0, t6
    slli a0, a0, 2
    lui  t0, 0x100
    add  a0, a0, t0

    /* B地址 */
    mul  a1, t6, t4
    add  a1, a1, t1
    slli a1, a1, 2
    lui  t0, 0x200
    add  a1, a1, t0

    /* 乘加 */
    flw  f1, 0(a0)
    flw  f2, 0(a1)
    fmadd.s f0, f1, f2, f0

    /* k++ */
    addi t6, t6, 1
    j    loop

done:
    /* C地址 */
    mul  a0, t2, t4
    add  a0, a0, t1
    slli a0, a0, 2
    lui  t0, 0x300
    add  a0, a0, t0

    fsw  f0, 0(a0)

exit:
    ebreak
    .align 2