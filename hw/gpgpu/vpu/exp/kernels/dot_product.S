/*
 * dot_product.S — 浮点向量内积
 *   result = Σ A[i] * B[i]
 *
 * 单线程串行归约，测浮点乘加吞吐
 *
 * 内存: N@0x0, A[N]@0x100000, B[N]@0x200000
 * 输出: result(float)@0x4
 */
.text
.globl _start
_start:
    lw   t0, 0(zero)       /* t0 = N */
    lw   t1, 0x80(zero)          /* A base */
    lw   t2, 0x84(zero)          /* B base */

    fcvt.s.w f0, zero       /* f0 = sum = 0.0 */
    addi t3, zero, 0        /* i = 0 */

loop:
    bge  t3, t0, done

    slli t4, t3, 2
    add  t4, t4, t1
    flw  ft2, 0(t4)         /* A[i] — ft2, not ft0! */

    slli t4, t3, 2
    add  t4, t4, t2
    flw  ft1, 0(t4)         /* B[i] */

    fmadd.s f0, ft2, ft1, f0 /* sum += A[i] * B[i] */

    addi t3, t3, 1
    jal  zero, loop

done:
    fsw  f0, 4(zero)        /* store result */
    ebreak
