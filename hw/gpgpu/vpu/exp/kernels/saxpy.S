/*
 * saxpy.S — 单精度浮点 SAXPY
 *
 * 单线程串行: for i in 0..N-1: Y[i] = A * X[i] + Y[i]
 *
 * 内存布局:
 *   VRAM[0x0000]  = N (uint32_t, 元素个数)
 *   VRAM[0x0004]  = A (float32, 标量系数)
 *   VRAM[0x1000]  = X 数组 (float32[N])
 *   VRAM[0x2000]  = Y 数组 (float32[N], 既是输入也是输出)
 *
 * 编译:
 *   riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32 -nostdlib -Ttext=0x0 -o s.out saxpy.S
 *   riscv64-unknown-elf-objcopy -O binary -j .text s.out saxpy.bin
 */

.text
.globl _start
_start:
    /* 加载 N */
    lw   t0, 0(zero)       /* t0 = N */

    /* 加载标量 A (float) */
    lw   t1, 4(zero)       /* t1 = A (位模式) */
    fmv.w.x ft0, t1         /* ft0 = A (float) */

    /* 基地址 */
    lui  t1, 0x100          /* t1 = 0x100000 (X 基地址) */
    lui  t2, 0x200          /* t2 = 0x200000 (Y 基地址) */

    /* i = 0 */
    addi t3, zero, 0        /* t3 = i */

loop:
    bge  t3, t0, done       /* if i >= N, goto done */

    /* 加载 X[i] */
    slli t4, t3, 2          /* t4 = i * 4 */
    add  t4, t4, t1         /* t4 = &X[i] */
    flw  ft1, 0(t4)         /* ft1 = X[i] */

    /* 加载 Y[i] */
    slli t4, t3, 2
    add  t4, t4, t2         /* t4 = &Y[i] */
    flw  ft2, 0(t4)         /* ft2 = Y[i] */

    /* Y[i] = A * X[i] + Y[i] */
    fmul.s ft3, ft0, ft1    /* ft3 = A * X[i] */
    fadd.s ft2, ft3, ft2    /* ft2 = A * X[i] + Y[i] */

    /* 存储 Y[i] */
    slli t4, t3, 2
    add  t4, t4, t2
    fsw  ft2, 0(t4)

    /* i++ */
    addi t3, t3, 1
    jal  zero, loop

done:
    ebreak
