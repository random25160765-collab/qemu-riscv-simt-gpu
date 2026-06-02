/*
 * vadd_vl.S — 测试 vsetvli + VL 控制
 * 只处理前 VL 个元素，剩余保持原值
 */
	.section .text
	.globl _start

_start:
	/* 从 ptr_table 加载 buffer 基地址 */
	lw   a0, 0x80(zero)         # A base
	lw   a1, 0x84(zero)         # B base
	lw   a2, 0x88(zero)         # C base

	/* VL = 16 (只处理前 16 个元素) */
	addi a3, zero, 16
	vsetvli a4, a3

	/* 加载 16 个 int32 */
	vle32.v v1, (a0)
	vle32.v v2, (a1)

	vadd.vv v3, v1, v2

	/* 存储结果（只有前 16 个元素被写入） */
	vse32.v v3, (a2)

	ebreak
	.align 2
