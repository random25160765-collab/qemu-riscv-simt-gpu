/*
 * soa.h — AoS ↔ SoA marshalling
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef SOA_H
#define SOA_H

#include <stdint.h>
#include "gpgpu_core.h"

static inline void aos_to_soa(const GPGPUWarp *warp, uint32_t gpr[32 * 32], uint32_t fpr[32 * 32], uint32_t *vpr,
                              uint32_t pc[32], uint32_t mhartid[32], uint32_t fcsr[32])
{
    for (int lane = 0; lane < 32; lane++) {
        const GPGPULane *l = &warp->lanes[lane];
        pc[lane] = l->pc;
        mhartid[lane] = l->mhartid;
        fcsr[lane] = l->fcsr;
        for (int r = 0; r < GPGPU_NUM_REGS; r++) {
            gpr[r * 32 + lane] = l->gpr[r].u32;
            fpr[r * 32 + lane] = l->fpr[r].u32;
            if (vpr) vpr[r * 32 + lane] = l->vpr[r].u32;
        }
    }
}

static inline void soa_to_aos(GPGPUWarp *warp, const uint32_t gpr[32 * 32], const uint32_t fpr[32 * 32],
                              const uint32_t *vpr, const uint32_t pc[32], const uint32_t mhartid[32],
                              const uint32_t fcsr[32])
{
    for (int lane = 0; lane < 32; lane++) {
        GPGPULane *l = &warp->lanes[lane];
        l->pc = pc[lane];
        l->mhartid = mhartid[lane];
        l->fcsr = fcsr[lane];
        for (int r = 0; r < GPGPU_NUM_REGS; r++) {
            l->gpr[r].u32 = gpr[r * 32 + lane];
            l->fpr[r].u32 = fpr[r * 32 + lane];
            if (vpr) l->vpr[r].u32 = vpr[r * 32 + lane];
        }
    }
}

#endif /* SOA_H */
