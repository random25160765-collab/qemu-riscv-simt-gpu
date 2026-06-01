/*
 * 矩阵乘法 GPU Kernel (标准 CUDA grid/block 模式)
 * C = A * B  (M x K) * (K x N) = (M x N)
 *
 * 参数: K @ VRAM[0], A[M][K]@0x100000, B[K][N]@0x200000, C[M][N]@0x300000
 * grid_dim[0] = M, block_dim[0] = N
 * blockIdx.x = row, threadIdx.x = col
 */

.section .text
.globl _start

_start:
    /* 读取 K = VRAM[0] */
    lw   t5, 0(zero)

    /* threadIdx.x = col */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* blockIdx.x = row */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)

    /* M = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)

    /* N = block_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t4, 0(t0)

    /* 边界检查 */
    bge  t2, t3, exit
    bge  t1, t4, exit

    /* sum = 0.0 */
    fcvt.s.w f0, zero

    /* k = 0 */
    addi t6, zero, 0

loop:
    bge  t6, t5, done

    /* A地址 = row*K + k */
    mul  a0, t2, t5
    add  a0, a0, t6
    slli a0, a0, 2
    lui  t0, 0x100
    add  a0, a0, t0

    /* B地址 = k*N + col */
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
    /* C地址 = row*N + col */
    mul  a0, t2, t4
    add  a0, a0, t1
    slli a0, a0, 2
    lui  t0, 0x300
    add  a0, a0, t0

    fsw  f0, 0(a0)

exit:
    ebreak
    .align 2