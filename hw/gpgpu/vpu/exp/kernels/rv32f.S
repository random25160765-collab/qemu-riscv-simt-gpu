/*
 * rv32f.S — RV32F 单精度浮点全覆盖测试
 *
 * 测试指令: FADD/FSUB/FMUL/FDIV/FSQRT, FMA(FMADD/FMSUB/FNMSUB/FNMADD),
 *            FSGNJ/FSGNJN/FSGNJX, FMIN/FMAX,
 *            FEQ/FLT/FLE, FCVT.W.S/FCVT.WU.S/FCVT.S.W/FCVT.S.WU,
 *            FMV.W.X/FMV.X.W, FCLASS
 *
 * 输入: VRAM[0x100000] = {a(flt), b(flt), c(flt), i32_val, u32_val, ...}
 *       其中 a=3.0f, b=2.0f, c=4.0f, i32_val=-5, u32_val=7
 *
 * 输出: VRAM[0x300000] = 结果数组 (按顺序排列)
 *
 * 内存布局:
 *   VRAM[0x100000] = 输入: {3.0f, 2.0f, 4.0f, -5(i32), 7(u32)}
 *   VRAM[0x300000] = 输出: 25 个 float32/uint32 结果
 *   VRAM[0x400000] = kernel 代码
 */

.text
.globl _start
_start:
    /* ================================================================
     * 加载测试数据到寄存器
     * ================================================================ */
    lw   t0, 0x80(zero)          /* t0 = 0x100000 (输入基地址) */

    /* 加载 a=3.0f, b=2.0f, c=4.0f */
    flw  ft0, 0(t0)         /* ft0 = a = 3.0f */
    flw  ft1, 4(t0)         /* ft1 = b = 2.0f */
    flw  ft2, 8(t0)         /* ft2 = c = 4.0f */

    /* 加载整数测试值: i32=-5, u32=7 */
    lw   t1, 12(t0)         /* t1 = -5 (i32) */
    lw   t2, 16(t0)         /* t2 = 7 (u32) */

    /* 输出基地址 */
    lw   t3, 0x88(zero)          /* t3 = 0x300000 */

    /* ================================================================
     * 基本算术: fadd, fsub, fmul, fdiv, fsqrt
     * ================================================================ */
    fadd.s   ft3, ft0, ft1      /* ft3 = 3.0 + 2.0 = 5.0 */
    fsw      ft3, 0(t3)

    fsub.s   ft3, ft0, ft1      /* ft3 = 3.0 - 2.0 = 1.0 */
    fsw      ft3, 4(t3)

    fmul.s   ft3, ft0, ft1      /* ft3 = 3.0 * 2.0 = 6.0 */
    fsw      ft3, 8(t3)

    fdiv.s   ft3, ft0, ft1      /* ft3 = 3.0 / 2.0 = 1.5 */
    fsw      ft3, 12(t3)

    fsqrt.s  ft3, ft0           /* ft3 = sqrt(3.0) ≈ 1.732 */
    fsw      ft3, 16(t3)

    /* ================================================================
     * FMA: fmadd, fmsub, fnmsub, fnmadd
     * fmadd:  d = a*b + c
     * fmsub:  d = a*b - c
     * fnmsub: d = -(a*b - c) = c - a*b
     * fnmadd: d = -(a*b + c)
     * ================================================================ */
    fmadd.s  ft3, ft0, ft1, ft2   /* ft3 = 3.0*2.0 + 4.0 = 10.0 */
    fsw      ft3, 20(t3)

    fmsub.s  ft3, ft0, ft1, ft2   /* ft3 = 3.0*2.0 - 4.0 = 2.0 */
    fsw      ft3, 24(t3)

    fnmsub.s ft3, ft0, ft1, ft2   /* ft3 = -(3.0*2.0 - 4.0) = -2.0 */
    fsw      ft3, 28(t3)

    fnmadd.s ft3, ft0, ft1, ft2   /* ft3 = -(3.0*2.0 + 4.0) = -10.0 */
    fsw      ft3, 32(t3)

    /* ================================================================
     * 符号注入: fsgnj, fsgnjn, fsgnjx
     * ================================================================ */
    fsgnj.s  ft3, ft0, ft1    /* ft3 = |3.0| with sign of 2.0 = 3.0 */
    fsw      ft3, 36(t3)

    fsgnjn.s ft3, ft0, ft1    /* ft3 = |3.0| with -sign of 2.0 = -3.0 */
    fsw      ft3, 40(t3)

    fsgnjx.s ft3, ft0, ft1    /* ft3 = 3.0 xor 2.0 sign = 3.0 (both +) */
    fsw      ft3, 44(t3)

    /* ================================================================
     * 最值: fmin, fmax
     * ================================================================ */
    fmin.s   ft3, ft0, ft1    /* ft3 = min(3.0, 2.0) = 2.0 */
    fsw      ft3, 48(t3)

    fmax.s   ft3, ft0, ft1    /* ft3 = max(3.0, 2.0) = 3.0 */
    fsw      ft3, 52(t3)

    /* ================================================================
     * 比较: feq, flt, fle
     * ================================================================ */
    feq.s    t4, ft0, ft1     /* t4 = (3.0 == 2.0) ? 1 : 0 = 0 */
    sw       t4, 56(t3)

    flt.s    t4, ft0, ft1     /* t4 = (3.0 < 2.0) ? 1 : 0 = 0 */
    sw       t4, 60(t3)

    fle.s    t4, ft1, ft0     /* t4 = (2.0 <= 3.0) ? 1 : 0 = 1 */
    sw       t4, 64(t3)

    flt.s    t4, ft1, ft0     /* t4 = (2.0 < 3.0) ? 1 : 0 = 1 */
    sw       t4, 68(t3)

    /* ================================================================
     * 转换: fcvt.w.s, fcvt.wu.s, fcvt.s.w, fcvt.s.wu
     * a=3.0f, b=2.0f, i32=-5, u32=7
     * ================================================================ */
    fcvt.w.s  t4, ft0         /* t4 = (int32_t)3.0f = 3 */
    sw        t4, 72(t3)

    fcvt.wu.s t4, ft0         /* t4 = (uint32_t)3.0f = 3 */
    sw        t4, 76(t3)

    fcvt.s.w  ft3, t1         /* ft3 = (float)(-5) = -5.0 */
    fsw       ft3, 80(t3)

    fcvt.s.wu ft3, t2         /* ft3 = (float)(7) = 7.0 */
    fsw       ft3, 84(t3)

    /* ================================================================
     * 数据移动: fmv.w.x, fmv.x.w
     * ================================================================ */
    fmv.w.x   ft3, t2         /* ft3 = bits of 7 as float (not 7.0f!) */
    fsw       ft3, 88(t3)

    fmv.x.w   t4, ft0         /* t4 = bits of 3.0f = 0x40400000 */
    sw        t4, 92(t3)

    /* ================================================================
     * 分类: fclass.s
     * fclass(3.0f): exp=0x80, mant!=0 → positive normal = bit 6
     * fclass(-3.0f): → negative normal = bit 1
     * ================================================================ */
    fclass.s  t4, ft0         /* t4 = class(3.0f) */
    sw        t4, 96(t3)

    /* fsgnjn.s(-3.0f) already gave us -3.0 in ft3 at offset 40 */
    flw       ft4, 40(t3)     /* ft4 = -3.0f */
    fclass.s  t4, ft4         /* t4 = class(-3.0f) */
    sw        t4, 100(t3)

    ebreak
