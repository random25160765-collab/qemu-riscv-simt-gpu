/*
 * Conv2D 卷积 Kernel (简化版：单通道，无 padding，stride=1)
 * 
 * 每个线程计算一个输出位置
 * 输入:  H x W 矩阵 (0x100000)
 * 卷积核: K x K 矩阵 (0x200000)
 * 输出:  (H-K+1) x (W-K+1) 矩阵 (0x300000)
 * 
 * 参数通过寄存器传递:
 * - grid_dim[0] = H (输入高度)
 * - grid_dim[1] = W (输入宽度)
 * - block_dim[0] = K (卷积核大小)
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
    
    /* 读取 K = block_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x20
    lw   t5, 0(t0)          # t5 = K
    
    /* 计算输出尺寸 */
    sub  t6, t3, t5
    addi t6, t6, 1          # out_h = H - K + 1
    sub  a0, t4, t5
    addi a0, a0, 1          # out_w = W - K + 1
    
    /* 边界检查 */
    bge  t2, t6, exit       # out_y >= out_h
    bge  t1, a0, exit       # out_x >= out_w
    
    /* 初始化累加器 sum = 0.0 */
    fcvt.s.w f0, zero
    
    /* 循环变量 ky = 0 */
    addi a1, zero, 0        # ky = 0
    
loop_ky:
    bge  a1, t5, done       # ky >= K
    
    /* 循环变量 kx = 0 */
    addi a2, zero, 0        # kx = 0
    
loop_kx:
    bge  a2, t5, next_ky    # kx >= K
    
    /* 计算输入地址: (out_y + ky) * W + (out_x + kx) */
    add  a3, t2, a1         # in_y = out_y + ky
    add  a4, t1, a2         # in_x = out_x + kx
    
    mul  a5, a3, t4
    add  a5, a5, a4
    slli a5, a5, 2
    lui  t0, 0x100
    add  a5, a5, t0
    
    /* 计算卷积核地址: (ky * K + kx) * 4 */
    mul  a6, a1, t5
    add  a6, a6, a2
    slli a6, a6, 2
    lui  t0, 0x200
    add  a6, a6, t0
    
    /* 加载并乘加 */
    flw  f1, 0(a5)          # f1 = input[in_y][in_x]
    flw  f2, 0(a6)          # f2 = kernel[ky][kx]
    fmadd.s f0, f1, f2, f0  # sum += f1 * f2
    
    addi a2, a2, 1          # kx++
    j    loop_kx
    
next_ky:
    addi a1, a1, 1          # ky++
    j    loop_ky
    
done:
    /* 计算输出地址: (out_y * out_w + out_x) * 4 */
    mul  a5, t2, a0
    add  a5, a5, t1
    slli a5, a5, 2
    lui  t0, 0x300
    add  a5, a5, t0
    
    /* 存储结果 */
    fsw  f0, 0(a5)
    
exit:
    ebreak
    .align 2