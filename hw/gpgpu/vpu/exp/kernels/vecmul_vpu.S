/*
 * vecmul_vpu.S — RVV 向量乘法: C[0..31] = A[0..31] * B[0..31]
 * 单 warp, 32 元素, RVV 1.0 标准语法
 */
	.section .text
	.globl _start

_start:
	lw   t1, 0x80(zero)             # A base
	vle32.v v0, (t1)

	lw   t1, 0x84(zero)             # B base
	vle32.v v1, (t1)

	vfmul.vv v2, v0, v1

	lw   t1, 0x88(zero)             # C base
	vse32.v v2, (t1)

	ebreak
	.align 2
