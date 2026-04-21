/*
 * MaxPool 2x2 池化 Kernel
 * 输入:  H x W 矩阵 (0x100000)
 * 输出:  H/2 x W/2 矩阵 (0x200000)
 * 
 * 每个线程处理一个输出位置
 * 参数:
 * - grid_dim[0] = H (输入高度)
 * - grid_dim[1] = W (输入宽度)
 */

.section .text
.globl _start

_start:
    /* 读取 thread_id = out_col */
    lui  t0, 0x80000
    lw   t1, 0(t0)          # t1 = out_x
    
    /* 读取 block_id = out_row */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)          # t2 = out_y
    
    /* 读取 H = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)          # t3 = H
    
    /* 读取 W = grid_dim[1] */
    lui  t0, 0x80000
    addi t0, t0, 0x34
    lw   t4, 0(t0)          # t4 = W
    
    /* 计算输出尺寸 */
    srli t5, t3, 1          # out_h = H / 2
    srli t6, t4, 1          # out_w = W / 2
    
    /* 边界检查 */
    bge  t2, t5, exit       # out_y >= out_h
    bge  t1, t6, exit       # out_x >= out_w
    
    /* 计算 2x2 窗口的起始位置 */
    slli a0, t2, 1          # in_y = out_y * 2
    slli a1, t1, 1          # in_x = out_x * 2
    
    /* 初始化最大值 = -inf */
    li   t0, 0xFF800000     # -inf 的 IEEE 754 表示
    fmv.w.x f0, t0
    
    /* 循环 y = 0, 1 */
    addi a2, zero, 0        # y = 0
    
loop_y:
    addi a3, zero, 0        # x = 0
    
loop_x:
    /* 计算地址: (in_y + y) * W + (in_x + x) */
    add  a4, a0, a2         # row = in_y + y
    add  a5, a1, a3         # col = in_x + x
    
    /* 边界检查 */
    bge  a4, t3, next_x
    bge  a5, t4, next_x
    
    /* 地址 = base + (row * W + col) * 4 */
    mul  a6, a4, t4
    add  a6, a6, a5
    slli a6, a6, 2
    lui  t0, 0x100
    add  a6, a6, t0
    
    /* 加载并比较 */
    flw  f1, 0(a6)
    fmax.s f0, f0, f1
    
next_x:
    addi a3, a3, 1
    addi t0, zero, 2
    blt  a3, t0, loop_x
    
    addi a2, a2, 1
    addi t0, zero, 2
    blt  a2, t0, loop_y
    
    /* 计算输出地址: base_out + (out_y * out_w + out_x) * 4 */
    mul  a6, t2, t6
    add  a6, a6, t1
    slli a6, a6, 2
    lui  t0, 0x200
    add  a6, a6, t0
    
    /* 存储最大值 */
    fsw  f0, 0(a6)
    
exit:
    ebreak
    .align 2