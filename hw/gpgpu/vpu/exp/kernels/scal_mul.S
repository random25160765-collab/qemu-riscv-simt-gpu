/*
 * scal_mul.S — 标准 RVV 标量乘向量: C[i] = alpha * V[i]
 * warp 级 SIMD。使用标准 RVV vle32.v / vfmul.vf / vse32.v
 */
	.section .text
	.globl _start

_start:
	lui  t0, 0x80000
	lw   t1, 0(t0)              # threadIdx.x

	addi t0, zero, -32
	and  t1, t1, t0             # col_start

	lui  t0, 0x80000
	addi t0, t0, 0x30
	lw   t3, 0(t0)              # gridDim.x
	lui  t0, 0x80000
	addi t0, t0, 0x20
	lw   t2, 0(t0)              # blockDim.x
	mul  t3, t3, t2             # total N

	bge  t1, t3, exit
	slli t1, t1, 2

	/* alpha from ptr_table[3] */
	lw   t5, 0x8C(zero)
	flw  f0, 0(t5)

	/* v0 = V[col_start:] */
	lw   t4, 0x80(zero)
	add  t4, t4, t1
	vle32.v v0, (t4)

	/* v1 = alpha * v0 */
	vfmul.vf v1, v0, f0

	/* store */
	lw   t4, 0x84(zero)
	add  t4, t4, t1
	vse32.v v1, (t4)

exit:
	ebreak
	.align 2
