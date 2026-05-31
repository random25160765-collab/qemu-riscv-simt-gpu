/*
 * vector_add.S — 整数向量加法
 *
 * 单线程串行: for i in 0..N-1: C[i] = A[i] + B[i]
 *
 * 内存布局:
 *   VRAM[0x0000]  = N (uint32_t, 元素个数)
 *   VRAM[0x1000]  = A 数组 (uint32_t[N])
 *   VRAM[0x2000]  = B 数组 (uint32_t[N])
 *   VRAM[0x3000]  = C 数组 (uint32_t[N], 输出)
 *   VRAM[0x10000] = kernel 代码
 *
 * 编译:
 *   riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -Ttext=0x0 -o v.out vector_add.S
 *   riscv64-unknown-elf-objcopy -O binary -j .text v.out vector_add.bin
 */

.text
.globl _start
_start:
    /* 加载 N */
    lw   t0, 0(zero)       /* t0 = N (从 VRAM[0x0] 读取) */

    /* 基地址 */
    lui  t1, 0x100          /* t1 = 0x100000 (A 基地址) */
    lui  t2, 0x200          /* t2 = 0x200000 (B 基地址) */
    lui  t3, 0x300          /* t3 = 0x300000 (C 基地址) */

    /* i = 0 */
    addi t4, zero, 0        /* t4 = i */

loop:
    bge  t4, t0, done       /* if i >= N, goto done */

    /* 加载 A[i] */
    slli t5, t4, 2          /* t5 = i * 4 */
    add  t5, t5, t1         /* t5 = &A[i] */
    lw   t6, 0(t5)          /* t6 = A[i] */

    /* 加载 B[i] */
    slli t5, t4, 2
    add  t5, t5, t2
    lw   a0, 0(t5)          /* a0 = B[i] */

    /* C[i] = A[i] + B[i] */
    add  a0, t6, a0

    /* 存储 C[i] */
    slli t5, t4, 2
    add  t5, t5, t3
    sw   a0, 0(t5)

    /* i++ */
    addi t4, t4, 1
    jal  zero, loop

done:
    ebreak
