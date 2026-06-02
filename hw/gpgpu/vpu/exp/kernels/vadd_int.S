/*
 * vadd_int.S — 标准 RVV 整数向量加法测试
 * 测试 vadd.vv (OPIVV, funct6=000000, vm=1)
 */
	.section .text
	.globl _start

_start:
	/* 从 ptr_table 加载 buffer 基地址 */
	lw   a0, 0x80(zero)         # A base = ptr_table[0]
	lw   a1, 0x84(zero)         # B base = ptr_table[1]
	lw   a2, 0x88(zero)         # C base = ptr_table[2]

	/* 加载 32 个 int32 (标准 RVV unit-stride) */
	vle32.v v1, (a0)            # v1 = A[0..31]
	vle32.v v2, (a1)            # v2 = B[0..31]

	vadd.vv v3, v1, v2

	/* 存储结果 */
	vse32.v v3, (a2)

	ebreak
	.align 2
