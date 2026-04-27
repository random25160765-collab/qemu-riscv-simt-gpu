.section .text
.globl _start

_start:
    /* 读取 MHARTID (CSR 0xF14) = wid*4 + tid */
    csrr t1, 0xF14         # t1 = global hart id (0..15)

    /* offset = global_id * 4 */
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

    /* 设置 exitcode = 0 并退出 */
    addi gp, zero, 0        # gp = x3 = 0 (return 0)
    ebreak

    .align 2