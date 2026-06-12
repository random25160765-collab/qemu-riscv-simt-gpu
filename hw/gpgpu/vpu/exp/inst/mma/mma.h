/*
 * mma.h — MMA 指令集入口
 *
 * engine.c 通过 #include 引入 MMA handler。
 * 当前为 hand-written stub, 指令集稳定后迁移到 spec+生成器管线。
 */

    /* ============================================================
     * TCU 矩阵指令 — warp-MMA (custom-1 opcode 0x2B, funct3=111)
     * mma 配置是 warp 本地变量, 避免多 warp 并发写入 s->mma 的 data race
     * ============================================================ */
op_mma_cfg: {
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    (void)rd;
    (void)rs1;
    (void)rs2;
    FOR_EACH_LANE PC(_li) += 4;
    NEXT();
}
op_mma_zero: {
    int acc = ip[-1].rd;
    memset(&_mma_acc[acc * 32], 0, 128);
    FOR_EACH_LANE PC(_li) += 4;
    NEXT();
}
op_mma_ld: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) _mma_acc[acc * 32 + _li] = *(float *)(s->vram_ptr + a);
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_s: {
    int acc = ip[-1].rd, vs1 = ip[-1].rs1, fs2 = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    float ak = FR(fs2, 0);
    FOR_EACH_LANE
    {
        _mma_acc[acc * 32 + _li] += VR(vs1, _li) * ak;
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_st: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) *(float *)(s->vram_ptr + a) = _mma_acc[acc * 32 + _li];
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_relu: {
    int acc = ip[-1].rd;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        float v = _mma_acc[acc * 32 + _li];
        _mma_acc[acc * 32 + _li] = v > 0 ? v : 0;
        PC(_li) += 4;
    }
    NEXT();
}
op_mma_bias: {
    int acc = ip[-1].rd, base = ip[-1].rs1, stride = ip[-1].rs2;
    if (UNLIKELY(acc >= 8)) return -1;
    FOR_EACH_LANE
    {
        uint32_t a = GPR(base, _li) + _li * GPR(stride, _li);
        if (LIKELY(a + 4 <= s->vram_size)) _mma_acc[acc * 32 + _li] += *(float *)(s->vram_ptr + a);
        PC(_li) += 4;
    }
    NEXT();
}
