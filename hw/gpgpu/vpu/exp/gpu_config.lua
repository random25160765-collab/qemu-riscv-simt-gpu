-- VPU Configuration
return {
    device = {
        num_cus      = 0,   -- 0 = auto (CPU cores)
        warps_per_cu = 1,   -- CU 内 warp 数
        warp_size    = 32,
        vram_mb      = 64,
    },
    features = {
        vpu    = true,   -- VPU vector accelerator
        tcu    = false,  -- TCU tensor accelerator (WIP)
        sfu    = true,   -- special function unit (exp/ln/sin/cos/...)
        lp     = true,   -- low-precision float (bf16/e4m3/e5m2/e2m1)
        debug  = false,  -- event output via ring buffer
        trace  = false,  -- instruction-level trace
        perf   = false,   -- hot-path stats: bandwidth, branches, divergence
    },
}
