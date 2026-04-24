/*
 * MaxPool 2x2 多通道 Kernel
 * 输入:  [C, H, W] @ 0x100000  (行优先: [c][y][x])
 * 输出:  [C, H/2, W/2] @ 0x200000
 *
 * Launch 参数:
 *   grid_x   = out_h = H/2      (blockIdx.x = out_y)
 *   grid_y   = C                (blockIdx.y = c)
 *   grid_z   = H                (grid_dim[2] 存 H，用于计算)
 *   block_x  = out_w = W/2      (threadIdx.x = out_x)
 *   block_dim[0] = W            (用 block_y 传 W)
 *
 * 线程映射:
 *   threadIdx.x = out_x
 *   blockIdx.x  = out_y
 *   blockIdx.y  = c
 */

.section .text
.globl _start

_start:
    /* threadIdx.x = out_x */
    lui  t0, 0x80000
    lw   t1, 0(t0)          # t1 = out_x

    /* blockIdx.x = out_y */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)          # t2 = out_y

    /* blockIdx.y = c */
    lui  t0, 0x80000
    addi t0, t0, 0x14
    lw   t3, 0(t0)          # t3 = c

    /* grid_dim[0] = out_h */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t4, 0(t0)          # t4 = out_h

    /* grid_dim[1] = C */
    lui  t0, 0x80000
    addi t0, t0, 0x34
    lw   t5, 0(t0)          # t5 = C

    /* grid_dim[2] = H */
    lui  t0, 0x80000
    addi t0, t0, 0x38
    lw   t6, 0(t0)          # t6 = H

    /* block_dim[0] = W */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   a0, 0(t0)          # a0 = W

    /* out_w = W / 2 */
    srli a1, a0, 1          # a1 = out_w

    /* 边界检查 */
    bge  t1, a1, exit       # out_x >= out_w
    bge  t2, t4, exit       # out_y >= out_h
    bge  t3, t5, exit       # c >= C

    /* 计算 2x2 窗口起始位置 */
    slli s0, t2, 1          # in_y = out_y * 2
    slli s1, t1, 1          # in_x = out_x * 2

    /* 初始化最大值 = -inf */
    li   t0, 0xFF800000
    fmv.w.x f0, t0

    /* HW = H * W */
    mul  s2, t6, a0         # s2 = H * W

    /* 循环 dy = 0, 1 */
    addi a2, zero, 0

loop_y:
    addi a3, zero, 0

loop_x:
    add  a4, s0, a2         # row = in_y + dy
    add  a5, s1, a3         # col = in_x + dx

    bge  a4, t6, next_x    # row >= H
    bge  a5, a0, next_x    # col >= W

    /* addr = 0x100000 + (c * H * W + row * W + col) * 4 */
    mul  s3, t3, s2         # c * H * W
    mul  s4, a4, a0         # row * W
    add  s4, s4, a5
    add  s3, s3, s4

    slli s3, s3, 2
    lui  t0, 0x100
    add  s3, s3, t0
    flw  f1, 0(s3)
    fmax.s f0, f0, f1

next_x:
    addi a3, a3, 1
    addi t0, zero, 2
    blt  a3, t0, loop_x

    addi a2, a2, 1
    addi t0, zero, 2
    blt  a2, t0, loop_y

    /* 输出地址: 0x200000 + (c * out_h * out_w + out_y * out_w + out_x) * 4 */
    mul  s3, t4, a1         # out_h * out_w
    mul  s4, t3, s3         # c * out_h * out_w
    mul  s5, t2, a1         # out_y * out_w
    add  s5, s5, t1         # + out_x
    add  s4, s4, s5

    slli s4, s4, 2
    lui  t0, 0x200
    add  s4, s4, t0
    fsw  f0, 0(s4)

exit:
    ebreak
    .align 2
