/*
 * fvec_add.S — 单精度浮点向量加法（最简单 FP 测试）
 *
 * 单线程: for i in 0..N-1: C[i] = A[i] + B[i]
 *
 * 内存布局:
 *   VRAM[0x000000]  = N (uint32_t)
 *   VRAM[0x100000]  = A 数组 (float32[N])
 *   VRAM[0x200000]  = B 数组 (float32[N])
 *   VRAM[0x300000]  = C 数组 (float32[N], 输出)
 */

.text
.globl _start
_start:
    /* 加载 N */
    lw   t0, 0(zero)       /* t0 = N */

    /* 基地址 */
    lui  t1, 0x100          /* t1 = 0x100000 (A 基地址) */
    lui  t2, 0x200          /* t2 = 0x200000 (B 基地址) */
    lui  t3, 0x300          /* t3 = 0x300000 (C 基地址) */

    /* i = 0 */
    addi t4, zero, 0        /* t4 = i */

loop:
    bge  t4, t0, done       /* if i >= N, goto done */

    /* 加载 A[i] */
    slli t5, t4, 2          /* t5 = i * 4 */
    add  t5, t5, t1         /* t5 = &A[i] */
    flw  ft0, 0(t5)         /* ft0 = A[i] */

    /* 加载 B[i] */
    slli t5, t4, 2
    add  t5, t5, t2         /* t5 = &B[i] */
    flw  ft1, 0(t5)         /* ft1 = B[i] */

    /* C[i] = A[i] + B[i] */
    fadd.s ft2, ft0, ft1    /* ft2 = A[i] + B[i] */

    /* 存储 C[i] */
    slli t5, t4, 2
    add  t5, t5, t3         /* t5 = &C[i] */
    fsw  ft2, 0(t5)

    /* i++ */
    addi t4, t4, 1
    jal  zero, loop

done:
    ebreak
