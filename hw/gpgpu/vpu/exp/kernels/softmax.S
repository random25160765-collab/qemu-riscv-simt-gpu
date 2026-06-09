/*
 * softmax.S — Softmax: y[i] = exp(x[i]) / sum(exp(x[j]))
 * 自定义指令 vfexp.v / vredsum.v (opcode 0x2B) — 无标准 RVV 等价
 */

.section .text
.globl _start

_start:
    /* threadIdx.x (per-lane) */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* warp_base = threadIdx.x & ~31 */
    addi t0, zero, -32
    and  t0, t1, t0

    /* N = total elements */
    lui  t5, 0x80000
    addi t5, t5, 0x30
    lw   t3, 0(t5)

    /* my global index in t1[lane] */
    bge  t1, t3, exit

    /* my_x = x[global_id] */
    slli t2, t1, 2
    lw   t4, 0x80(zero)
    add  t2, t4, t2
    flw  f0, 0(t2)
    fexp.s f0, f0

    /* sum = 0, k = warp_base */
    fcvt.s.w f1, zero

loop:
    bge  t0, t3, done

    slli t2, t0, 2
    lw   t4, 0x80(zero)
    add  t2, t4, t2
    vle32.v v0, (t2)

    vfexp.v v0, v0
    vredsum.v f2, v0
    fadd.s f1, f1, f2

    addi t0, t0, 32
    j    loop

done:
    fdiv.s f0, f0, f1
    slli t2, t1, 2
    lw   t4, 0x84(zero)
    add  t2, t4, t2
    fsw  f0, 0(t2)

exit:
    ebreak
    .align 2
