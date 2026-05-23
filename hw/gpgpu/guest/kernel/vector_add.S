/*
 * 向量加法 GPU Kernel (RISC-V 汇编) - 无 barrier 版本
 */

.section .text
.globl _start

_start:
    /* 读取 thread_id_x (0x80000000) */
    lui  t0, 0x80000        # t0 = 0x80000000
    lw   t1, 0(t0)          # t1 = threadIdx.x

    /* 读取 block_id_x (0x80000010) */
    lui  t0, 0x80000
    addi t0, t0, 0x10       # t0 = 0x80000010
    lw   t2, 0(t0)          # t2 = blockIdx.x

    /* 读取 N = grid_dim[0] */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)          # t3 = N

    /* 计算全局 ID = blockIdx.x * 256 + threadIdx.x */
    addi t4, zero, 256      # t4 = 256
    mul  t2, t2, t4
    add  t1, t2, t1

    /* 边界检查 */
    bge  t1, t3, exit

    /* 计算数组偏移: global_id * 4 */
    slli t1, t1, 2

    /* 加载 A[global_id] */
    lui  t0, 0x100          # t0 = 0x100000
    add  t0, t0, t1
    flw  f0, 0(t0)

    /* 加载 B[global_id] */
    lui  t0, 0x200          # t0 = 0x200000
    add  t0, t0, t1
    flw  f1, 0(t0)

    /* C = A + B */
    fadd.s f2, f0, f1

    /* 存储到 C[global_id] */
    lui  t0, 0x300          # t0 = 0x300000
    add  t0, t0, t1
    fsw  f2, 0(t0)

exit:
    ebreak

    /* 确保 4 字节对齐 */
    .align 2