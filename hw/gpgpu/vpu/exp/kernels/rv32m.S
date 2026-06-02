/*
 * rv32m.S — RV32M 乘除扩展全覆盖测试
 *
 * 输入:  VRAM[0x100000] = {a0, b0, a1, b1, ...}  (N 对 uint32_t)
 * 输出:  VRAM[0x300000] = 每条指令的结果数组
 *
 * 测试指令: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
 * 每个 test case 消耗: 2×4B 输入 + 8×4B 输出 = 40B
 *
 * 内存布局:
 *   VRAM[0x000000] = N (test case 数量)
 *   VRAM[0x100000] = 输入数组
 *   VRAM[0x300000] = 输出数组
 *   VRAM[0x400000] = kernel 代码
 */

.text
.globl _start
_start:
    /* 加载 N */
    lw   t0, 0(zero)       /* t0 = N (test case 数量) */

    /* 基地址 */
    lw   t1, 0x80(zero)          /* t1 = 0x100000 (输入基地址) */
    lw   t2, 0x88(zero)          /* t2 = 0x300000 (输出基地址) */

    /* i = 0 */
    addi t3, zero, 0        /* t3 = i */

loop:
    bge  t3, t0, done       /* if i >= N, done */

    /* 加载输入 a[i], b[i] */
    slli t4, t3, 3          /* t4 = i * 8 (2 words per test case) */
    add  t4, t4, t1         /* t4 = &input[i*2] */
    lw   t5, 0(t4)          /* t5 = a */
    lw   t6, 4(t4)          /* t6 = b */

    /* 计算输出基地址 for this test case */
    slli a0, t3, 5          /* a0 = i * 32 (8 words output per case) */
    add  a0, a0, t2         /* a0 = &output[i*8] */

    /* 1. MUL */
    mul  a1, t5, t6
    sw   a1, 0(a0)

    /* 2. MULH */
    mulh a1, t5, t6
    sw   a1, 4(a0)

    /* 3. MULHSU */
    mulhsu a1, t5, t6
    sw   a1, 8(a0)

    /* 4. MULHU */
    mulhu a1, t5, t6
    sw   a1, 12(a0)

    /* 5. DIV (by zero → -1) */
    div  a1, t5, t6
    sw   a1, 16(a0)

    /* 6. DIVU (by zero → 0xFFFFFFFF) */
    divu a1, t5, t6
    sw   a1, 20(a0)

    /* 7. REM (by zero → a) */
    rem  a1, t5, t6
    sw   a1, 24(a0)

    /* 8. REMU (by zero → a) */
    remu a1, t5, t6
    sw   a1, 28(a0)

    /* i++ */
    addi t3, t3, 1
    jal  zero, loop

done:
    ebreak
