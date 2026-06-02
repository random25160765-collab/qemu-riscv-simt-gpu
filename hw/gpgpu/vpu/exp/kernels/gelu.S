/*
 * gelu.S — custom VPU GELU: out[i] = x * sigmoid(1.702 * x)
 * 使用 custom VPU (opcode 0x2B): vld.v / vfmul.vs / vfsig.v / vfmul.v / vst.v
 * 注: vfsig.v 无标准 RVV 等价指令，暂时保留 custom VPU 实现
 */
	.section .text
	.globl _start

_start:
	lui  t0, 0x80000
	lw   t1, 0(t0)

	addi t0, zero, -32
	and  t1, t1, t0

	lui  t0, 0x80000
	addi t0, t0, 0x30
	lw   t3, 0(t0)
	lui  t0, 0x80000
	addi t0, t0, 0x20
	lw   t2, 0(t0)
	mul  t3, t3, t2

	bge  t1, t3, exit
	slli t1, t1, 2

	/* alpha = 1.702f */
	lui  t5, 0x3FD9D
	addi t5, t5, -292
	fmv.w.x f0, t5

	addi t0, zero, 4

	/* v0 = x */
	lw   t4, 0x80(zero)
	add  t4, t4, t1
	vld.v v0, (t4), t0

	/* v1 = 1.702 * x */
	vfmul.vs v1, v0, f0
	/* v1 = sigmoid(1.702*x) */
	vfsig.v v1, v1
	/* v0 = x * sigmoid(1.702*x) */
	vfmul.v v0, v0, v1

	/* store */
	lw   t4, 0x84(zero)
	add  t4, t4, t1
	vst.v v0, (t4), t0

exit:
	ebreak
	.align 2
