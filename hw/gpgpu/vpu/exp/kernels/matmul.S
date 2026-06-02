/*
 * matmul.S — TCU 矩阵乘法, K-loop 4x 展开 + 累加器
 * mma.zero / mma.s / mma.st + vld.v
 */

.section .text
.globl _start

_start:
    lw   t5, 0(zero)            # K = VRAM[0]

    lui  t0, 0x80000
    lw   t1, 0(t0)              # threadIdx.x
    addi t0, zero, -32
    and  t1, t1, t0             # col_start = threadIdx.x & ~31

    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)              # row = blockIdx.x

    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t4, 0(t0)              # N = blockDim.x

    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)              # M = gridDim.x

    bge  t2, t3, exit
    bge  t1, t4, exit

    mma.zero acc0

    mul  a0, t2, t5
    slli a0, a0, 2
    lui  a3, 0x100
    add  a0, a0, a3             # a0 = A[row][0]

    addi a5, zero, 4            # stride = 4
    slli a6, t4, 2              # a6 = N * 4

    slli a2, t1, 2
    lui  a4, 0x200
    add  a2, a2, a4             # a2 = B[k][col_start]

    andi t6, t5, -4             # K rounded down to 4
    addi a7, zero, 0            # k counter

kloop4:
    bge  a7, t6, cleanup

    flw  f0, 0(a0)
    vld.v v1, (a2), a5
    mma.s acc0, v1, f0

    flw  f0, 4(a0)
    add  a2, a2, a6
    vld.v v1, (a2), a5
    mma.s acc0, v1, f0

    flw  f0, 8(a0)
    add  a2, a2, a6
    vld.v v1, (a2), a5
    mma.s acc0, v1, f0

    flw  f0, 12(a0)
    add  a2, a2, a6
    vld.v v1, (a2), a5
    mma.s acc0, v1, f0

    addi a0, a0, 16
    add  a2, a2, a6
    addi a7, a7, 4
    j    kloop4

cleanup:
    bge  a7, t5, done
    flw  f0, 0(a0)
    vld.v v1, (a2), a5
    mma.s acc0, v1, f0
    addi a0, a0, 4
    add  a2, a2, a6
    addi a7, a7, 1
    j    cleanup

done:
    mul  a2, t2, t4
    add  a2, a2, t1
    slli a2, a2, 2
    lui  t0, 0x300
    add  a2, a2, t0             # a2 = C[row][col_start]
    addi a5, zero, 4
    mma.st acc0, (a2), a5

exit:
    ebreak
    .align 2
