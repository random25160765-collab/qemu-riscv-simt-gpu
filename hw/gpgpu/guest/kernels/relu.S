/*
 * ReLU 激活函数 Kernel
 * f(x) = max(0, x)
 * 
 * 每个线程处理一个元素
 * 输入: 0x100000 (N 个 float)
 * 输出: 0x100000 (原地修改)
 * 
 * 参数通过寄存器传递:
 * - grid_dim[0] = N (元素数量)
 */

.section .text
.globl _start

_start:
    /* 读取 thread_id = 全局索引 */
    lui  t0, 0x80000
    lw   t1, 0(t0)          # t1 = threadIdx.x
    
    /* 读取 block_id */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)          # t2 = blockIdx.x
    
    /* 读取 N = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)          # t3 = N
    
    /* 计算全局 ID = blockIdx.x * 256 + threadIdx.x */
    addi t4, zero, 256
    mul  t2, t2, t4
    add  t1, t2, t1         # t1 = global_id
    
    /* 边界检查 */
    bge  t1, t3, exit
    
    /* 计算地址: base + global_id * 4 */
    slli t1, t1, 2
    lui  t0, 0x100
    add  t0, t0, t1         # t0 = &x[global_id]
    
    /* 加载数据 */
    flw  f0, 0(t0)          # f0 = x
    
    /* ReLU: f0 = max(f0, 0.0) */
    fcvt.s.w f1, zero       # f1 = 0.0
    fmax.s f0, f0, f1       # f0 = max(x, 0)
    
    /* 存储结果 */
    fsw  f0, 0(t0)
    
exit:
    ebreak
    .align 2