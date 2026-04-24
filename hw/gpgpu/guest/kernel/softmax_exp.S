.section .text
.globl _start

_start:
    /* 读取 thread_id */
    lui  t0, 0x80000
    lw   t1, 0(t0)          # t1 = threadIdx.x
    
    /* 读取 block_id */
    lui  t0, 0x80000
    addi t0, t0, 0x10
    lw   t2, 0(t0)          # t2 = blockIdx.x
    
    /* 读取 N */
    lui  t0, 0x80000
    addi t0, t0, 0x30
    lw   t3, 0(t0)          # t3 = N
    
    /* 计算 global_id = blockIdx.x * 256 + threadIdx.x，保存到 t4 */
    addi t4, zero, 256
    mul  t2, t2, t4
    add  t4, t2, t1         # t4 = global_id
    
    /* 边界检查 */
    bge  t4, t3, exit
    
    /* 加载输入 */
    slli t0, t4, 2
    lui  t1, 0x100
    add  t0, t1, t0
    flw  f0, 0(t0)          # f0 = x
    
    /* 常数: 1/ln2 ≈ 1.44269502 */
    li   t0, 0x3fb8aa3b
    fmv.w.x f1, t0
    
    /* 常数: ln2 ≈ 0.69314718 */
    li   t0, 0x3f317218
    fmv.w.x f2, t0
    
    /* k = round(x * 1.442695) */
    fmul.s f3, f0, f1
    fcvt.w.s t0, f3, rmm    # t0 = k (四舍五入)
    
    /* r = x - k * 0.693147 */
    fcvt.s.w f4, t0
    fmul.s f4, f4, f2
    fsub.s f0, f0, f4       # f0 = r
    
    /* 保存符号 */
    fmv.x.w t1, f0
    srli t2, t1, 31         # t2 = sign bit (1 if negative)
    
    /* 取绝对值 */
    li   t3, 0x7fffffff
    and  t1, t1, t3
    fmv.w.x f0, t1          # f0 = |r|
    
    /* 常数 1.0, 2.0, 6.0, 24.0, 120.0, 720.0 */
    addi t1, zero, 1
    fcvt.s.w f5, t1         # f5 = 1.0
    addi t1, zero, 2
    fcvt.s.w f6, t1         # f6 = 2.0
    addi t1, zero, 6
    fcvt.s.w f7, t1         # f7 = 6.0
    addi t1, zero, 24
    fcvt.s.w f8, t1         # f8 = 24.0
    addi t1, zero, 120
    fcvt.s.w f16, t1        # f16 = 120.0
    addi t1, zero, 720
    fcvt.s.w f17, t1        # f17 = 720.0
    
    /* 计算 r², r³, r⁴, r⁵, r⁶ */
    fmul.s f9, f0, f0       # f9 = r²
    fmul.s f10, f9, f0      # f10 = r³
    fmul.s f11, f10, f0     # f11 = r⁴
    fmul.s f18, f11, f0     # f18 = r⁵
    fmul.s f19, f18, f0     # f19 = r⁶
    
    /* 计算各项 */
    fdiv.s f12, f9, f6      # r²/2
    fdiv.s f13, f10, f7     # r³/6
    fdiv.s f14, f11, f8     # r⁴/24
    fdiv.s f20, f18, f16    # r⁵/120
    fdiv.s f21, f19, f17    # r⁶/720
    
    /* exp(|r|) = 1 + |r| + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 */
    fadd.s f15, f5, f0      # 1 + |r|
    fadd.s f15, f15, f12
    fadd.s f15, f15, f13
    fadd.s f15, f15, f14
    fadd.s f15, f15, f20
    fadd.s f15, f15, f21
    
    /* 如果原 r < 0，取倒数 */
    beqz t2, apply_k
    fdiv.s f15, f5, f15     # 1/exp(|r|)
    
apply_k:
    /* exp(x) = exp(r) * 2^k */
    fmv.x.w t1, f15
    slli t0, t0, 23
    add  t1, t1, t0
    fmv.w.x f15, t1
    
store:
    /* 存储结果到 0x200000 + global_id * 4 */
    slli t0, t4, 2
    lui  t1, 0x200
    add  t0, t1, t0
    fsw  f15, 0(t0)
    
exit:
    ebreak
    .align 2