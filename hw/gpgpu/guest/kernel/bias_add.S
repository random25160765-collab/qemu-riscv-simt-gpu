/*
 * BiasAdd Broadcast Kernel
 * out[i] = in[i] + bias[i / HW]
 *
 * 输入: [N] @ 0x100000  (N = C * H * W)
 * 偏置: [C] @ 0x200000
 * 输出: [N] @ 0x300000
 *
 * Launch 参数:
 *   grid_x   = ceil(N / 256)
 *   grid_y   = N              (grid_dim[1] 存 N)
 *   grid_z   = HW             (grid_dim[2] 存 HW = H * W)
 *   block_x  = 256
 *
 * 线程映射:
 *   global_id = blockIdx.x * 256 + threadIdx.x
 *   c         = global_id / HW
 */

.section .text
.globl _start

_start:
    /* threadIdx.x */
    lui  t0, 0x80000
    lw   t1, 0(t0)

    /* blockIdx.x */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)

    /* N = grid_dim[1] */
    lui  t0, 0x80000
    addi t0, t0, 0x34
    lw   t3, 0(t0)          # t3 = N

    /* HW = grid_dim[2] */
    lui  t0, 0x80000
    addi t0, t0, 0x38
    lw   t4, 0(t0)          # t4 = HW

    /* global_id = blockIdx.x * 256 + threadIdx.x */
    addi t5, zero, 256
    mul  t2, t2, t5
    add  t1, t2, t1         # t1 = global_id

    bge  t1, t3, exit

    /* c = global_id / HW */
    div  t5, t1, t4         # t5 = c

    /* 读 in[global_id] */
    slli t6, t1, 2
    lui  t0, 0x100
    add  t0, t0, t6
    flw  f0, 0(t0)

    /* 读 bias[c] */
    slli t5, t5, 2
    lui  t0, 0x200
    add  t0, t0, t5
    flw  f1, 0(t0)

    fadd.s f0, f0, f1

    /* 写 out[global_id] */
    slli t1, t1, 2
    lui  t0, 0x300
    add  t0, t0, t1
    fsw  f0, 0(t0)

exit:
    ebreak
    .align 2
