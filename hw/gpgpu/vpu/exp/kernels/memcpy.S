/*
 * memcpy.S — 批量内存拷贝（测访存带宽）
 *   dst[i] = src[i]   for i in 0..N-1
 *
 * 内存: N@0x0, src[N]@0x100000, dst[N]@0x200000
 */
.text
.globl _start
_start:
    lw   t0, 0(zero)       /* N */
    lui  t1, 0x100          /* src */
    lui  t2, 0x200          /* dst */
    addi t3, zero, 0        /* i */

loop:
    bge  t3, t0, done
    slli t4, t3, 2
    add  t4, t4, t1
    lw   t5, 0(t4)         /* load src[i] */
    slli t4, t3, 2
    add  t4, t4, t2
    sw   t5, 0(t4)         /* store dst[i] */
    addi t3, t3, 1
    jal  zero, loop

done:
    ebreak
