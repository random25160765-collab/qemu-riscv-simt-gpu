.section .text
.globl _start

_start:
    /* 读取 MHARTID (CSR 0xF14) = wid*4 + tid */
    csrr t1, 0xF14         # t1 = global hart id (0..15)

    /* offset = global_id * 4 */
    slli t1, t1, 2

    /* 加载 A[global_id] */
    flw  f0, 0(t1)         # A 在地址 0x0 + offset

    /* 加载 B[global_id] */
    lui  t0, 0x1           # t0 = 0x1000
    add  t0, t0, t1
    flw  f1, 0(t0)

    /* C = A + B */
    fadd.s f2, f0, f1

    /* 存储到 C[global_id] */
    lui  t0, 0x2           # t0 = 0x2000
    add  t0, t0, t1
    fsw  f2, 0(t0)

    /* 完成 */
    ebreak

    .align 2