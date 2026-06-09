/* ============================================================
 * GPU 控制指令 (手写)
 * ============================================================ */

/* ebreak / done — SIMT stack 感知出口 */
op_ebreak:
{
    _active = 0;
    if (_sdepth > 0) {
        _sdepth--;
        _active = _stk[_sdepth].mask;
        ip = &code[_stk[_sdepth].ft_idx];
        NEXT();
    }
    return 0;
}
op_done:
{
    _active = 0;
    if (_sdepth > 0) {
        _sdepth--;
        _active = _stk[_sdepth].mask;
        ip = &code[_stk[_sdepth].ft_idx];
        NEXT();
    }
    return 0;
}

/* barrier — warp 同步点, SIMT stack 写回 + return resume PC */
op_barrier:
{
    s->simt.barrier_active = 1;
    memcpy(_stk_ext, _stk, sizeof(_stk));
    *_sdepth_ext = _sdepth;
    return (int)(ip - code) | 0x10000;
}

/* tex — bilinear texture fetch */
op_tex:
{
    int rd = ip[-1].rd, rs1 = ip[-1].rs1, rs2 = ip[-1].rs2;
    uint32_t base = (uint32_t)ip[-1].imm;
    FOR_EACH_LANE
    {
        float u = fabsf(FR(rs1, _li)), v = fabsf(FR(rs2, _li));
        int iu = (int)u, iv = (int)v;
        if (iu < 0) iu = 0;
        if (iu > 254) {
            iu = 254;
            s->error_status |= GPGPU_ERR_VRAM_FAULT;
        }
        int max_row = (int)((s->vram_size - base) / (256 * 4)) - 1;
        if (max_row < 1) max_row = 1;
        if (iv < 0) iv = 0;
        if (iv >= max_row) {
            iv = max_row - 1;
            s->error_status |= GPGPU_ERR_VRAM_FAULT;
        }
        float fu = u - (float)iu, fv = v - (float)iv;
        float *tex = (float *)(s->vram_ptr + base);
        float s00 = tex[iv * 256 + iu], s10 = tex[iv * 256 + iu + 1];
        float s01 = tex[(iv + 1) * 256 + iu], s11 = tex[(iv + 1) * 256 + iu + 1];
        FR(rd, _li) = (1 - fu) * (1 - fv) * s00 + fu * (1 - fv) * s10 + (1 - fu) * fv * s01 + fu * fv * s11;
        PC(_li) += 4;
    }
    NEXT();
}