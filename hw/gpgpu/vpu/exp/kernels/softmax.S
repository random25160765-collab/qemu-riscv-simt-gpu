/*
 * softmax.S — VPU Softmax: y[i] = exp(x[i]) / sum(exp(x[j]))
 * VPU: vld.v/vfexp.v/vredsum.v 批量, warp 级
 */

.section .text
.globl _start

_start:
    /* threadIdx.x (per-lane) */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* my_idx = threadIdx.x & ~31 (warp base) + lane offset */
    /* For softmax, each element is independent output; we need the actual thread index */
    /* But the reduction loop needs to start from warp base */
    addi t0, zero, -32
    and  t0, t1, t0           /* t0 = warp_base (same for all lanes) */

    /* N = total elements */
    lui  t5, 0x80000
    addi t5, t5, 0x30
    lw   t3, 0(t5)

    /* my global index is in t1[lane] */
    bge  t1, t3, exit         /* per-lane boundary check */

    /* my_x = x[global_id] */
    slli t2, t1, 2
    lui  t4, 0x100
    add  t2, t4, t2
    flw  f0, 0(t2)
    fexp.s f0, f0

    /* sum = 0, k = warp_base */
    fcvt.s.w f1, zero
    addi t6, zero, 4

loop:
    bge  t0, t3, done

    slli t2, t0, 2
    lui  t4, 0x100
    add  t2, t4, t2
    vld.v v0, (t2), t6

    vfexp.v v0, v0
    vredsum.v f2, v0
    fadd.s f1, f1, f2

    addi t0, t0, 32
    j    loop

done:
    fdiv.s f0, f0, f1
    slli t2, t1, 2
    lui  t4, 0x200
    add  t2, t4, t2
    fsw  f0, 0(t2)

exit:
    ebreak
    .align 2
