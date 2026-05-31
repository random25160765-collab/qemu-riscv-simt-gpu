/*
 * fib.S — 斐波那契数列计算（测试分支密集型代码）
 *
 * 计算第 N 个斐波那契数（取模 2^32），存入 VRAM[0x4]
 *
 * 内存布局:
 *   VRAM[0x0000] = N (uint32_t, 要计算的项数)
 *   VRAM[0x0004] = fib(N) (uint32_t, 输出)
 *
 * 编译:
 *   riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -Ttext=0x0 -o f.out fib.S
 *   riscv64-unknown-elf-objcopy -O binary -j .text f.out fib.bin
 */

.text
.globl _start
_start:
    /* 加载 N */
    lw   t0, 0(zero)       /* t0 = N */

    /* 如果 N == 0，goto done_zero */
    beq  t0, zero, done_zero

    /* fib(0)=0, fib(1)=1 */
    addi t1, zero, 0        /* t1 = fib(n-2) */
    addi t2, zero, 1        /* t2 = fib(n-1) */

    /* 如果 N == 1，goto done_store */
    addi t3, zero, 1
    beq  t0, t3, done_store

    /* i = 2 */
    addi t3, zero, 2

loop:
    bltu t0, t3, done_store /* if N < i, done (i > N) */

    /* fib(i) = fib(i-1) + fib(i-2) */
    add  t4, t1, t2

    /* 移位 */
    addi t1, t2, 0
    addi t2, t4, 0

    /* i++ */
    addi t3, t3, 1
    jal  zero, loop

done_zero:
    addi t2, zero, 0
    sw   t2, 4(zero)
    ebreak

done_store:
    sw   t2, 4(zero)
    ebreak
