/*
 * Conv2D 多通道卷积 Kernel
 * 输入:  C_in x H x W @ 0x100000 (行优先: [in_c][y][x])
 * 权重:  C_out x C_in x K x K @ 0x200000 (行优先: [out_c][in_c][ky][kx])
 * 输出:  C_out x out_h x out_w @ 0x300000 (行优先: [out_c][y][x])
 * 
 * 参数:
 * - grid_dim[0] = H
 * - grid_dim[1] = W
 * - grid_dim[2] = C_in
 * - block_dim[0] = K
 * - block_dim[1] = C_out
 */

.section .text
.globl _start

_start:
    /* 读取 threadIdx.x = out_x */
    lui  t0, 0x80000
    lw   t1, 0(t0)          # t1 = out_x
    
    /* 读取 blockIdx.x = block_id */
    lui  t0, 0x80000
    addi  t0, t0, 0x10
    lw   t2, 0(t0)          # t2 = block_id
    
    /* 读取 H = grid_dim[0] */
    lui  t0, 0x80000
    addi  t0, t0, 0x30
    lw   t3, 0(t0)          # t3 = H
    
    /* 读取 W = grid_dim[1] */
    lui  t0, 0x80000
    addi  t0, t0, 0x34
    lw   t4, 0(t0)          # t4 = W
    
    /* 读取 C_in = grid_dim[2] */
    lui  t0, 0x80000
    addi  t0, t0, 0x38
    lw   t5, 0(t0)          # t5 = C_in
    
    /* 读取 K = block_dim[0] */
    lui  t0, 0x80000
    addi  t0, t0, 0x20
    lw   t6, 0(t0)          # t6 = K
    
    /* 读取 C_out = block_dim[1] */
    lui  t0, 0x80000
    addi  t0, t0, 0x24
    lw   a0, 0(t0)          # a0 = C_out
    
    /* 计算输出尺寸 */
    sub  a1, t3, t6         # out_h = H - K + 1
    addi a1, a1, 1
    sub  a2, t4, t6         # out_w = W - K + 1
    addi a2, a2, 1
    
    /* 解码 block_id -> (out_c, out_y) */
    div  a3, t2, a1         # out_y = block_id / out_h
    rem  a4, t2, a1         # out_c = block_id % out_h ? 
    /* 正确: out_c = block_id / out_h, out_y = block_id % out_h */
    div  a3, t2, a1         # a3 = out_c (block_id / out_h)
    rem  a4, t2, a1         # a4 = out_y (block_id % out_h)
    
    /* 边界检查 */
    bge  a3, a0, exit       # out_c >= C_out
    bge  a4, a1, exit       # out_y >= out_h
    bge  t1, a2, exit       # out_x >= out_w
    
    /* 初始化累加器 */
    fcvt.s.w f0, zero
    
    /* 预计算常量 */
    mul  s0, t5, t6         # C_in * K
    mul  s0, s0, t6         # C_in * K * K (每个输出通道的权重数)
    mul  s1, t3, t4         # H * W (每个输入通道的元素数)
    
    /* 循环 in_c = 0..C_in-1 */
    addi s2, zero, 0        # in_c = 0
    
loop_in_c:
    bge  s2, t5, done       # in_c >= C_in
    
    /* 循环 ky = 0..K-1 */
    addi s3, zero, 0        # ky = 0
    
loop_ky:
    bge  s3, t6, next_in_c  # ky >= K
    
    /* 循环 kx = 0..K-1 */
    addi s4, zero, 0        # kx = 0
    
loop_kx:
    bge  s4, t6, next_ky    # kx >= K
    
    /* === 输入地址 === */
    /* offset = in_c * H * W + (out_y + ky) * W + (out_x + kx) */
    mul  s5, s2, s1         # in_c * H * W
    add  s6, a4, s3         # out_y + ky
    mul  s6, s6, t4         # (out_y + ky) * W
    add  s6, s6, t1         # + out_x
    add  s6, s6, s4         # + kx
    add  s5, s5, s6         # 总偏移
    
    slli s5, s5, 2
    lui  s6, 0x100
    add  s5, s5, s6         # 输入地址
    
    /* === 权重地址 === */
    /* offset = out_c * C_in * K * K + in_c * K * K + ky * K + kx */
    mul  s7, a3, s0         # out_c * C_in * K * K
    mul  s8, s2, t6         # in_c * K
    mul  s8, s8, t6         # in_c * K * K
    mul  s9, s3, t6         # ky * K
    add  s9, s9, s4         # ky * K + kx
    add  s7, s7, s8
    add  s7, s7, s9         # 总偏移
    
    slli s7, s7, 2
    lui  s8, 0x200
    add  s7, s7, s8         # 权重地址
    
    /* 乘加 */
    flw  f1, 0(s5)
    flw  f2, 0(s7)
    fmadd.s f0, f1, f2, f0
    
    addi s4, s4, 1          # kx++
    j    loop_kx
    
next_ky:
    addi s3, s3, 1          # ky++
    j    loop_ky
    
next_in_c:
    addi s2, s2, 1          # in_c++
    j    loop_in_c
    
done:
    /* === 输出地址 === */
    /* offset = out_c * out_h * out_w + out_y * out_w + out_x */
    mul  s2, a1, a2         # out_h * out_w
    mul  s3, a3, s2         # out_c * out_h * out_w
    mul  s4, a4, a2         # out_y * out_w
    add  s4, s4, t1         # + out_x
    add  s3, s3, s4         # 总偏移
    
    slli s3, s3, 2
    lui  s4, 0x300
    add  s3, s3, s4         # 输出地址
    
    fsw  f0, 0(s3)
    
exit:
    ebreak
    .align 2