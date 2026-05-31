/*
 * simd_handlers.h — CSR/FP 辅助宏
 */
#ifndef SIMD_HANDLERS_H
#define SIMD_HANDLERS_H

#include <math.h>
#include "state.h"
#include "lpfp.h"
#include "softfloat/softfloat.h"

/* ============================================================
 * CSR helper
 * ============================================================ */
static inline uint32_t _csr_rd(GPGPULane *l, uint16_t a) {
    switch (a) {
        case CSR_MHARTID: return l->mhartid;
        case CSR_FFLAGS:  return l->fcsr & 0x1F;
        case CSR_FRM:     return (l->fcsr >> 5) & 0x7;
        case CSR_FCSR:    return l->fcsr;
        default: return 0;
    }
}
static inline void _csr_wr(GPGPULane *l, uint16_t a, uint32_t v) {
    switch (a) {
        case CSR_FFLAGS: l->fcsr = (l->fcsr & ~0x1F) | (v & 0x1F); break;
        case CSR_FRM:    l->fcsr = (l->fcsr & ~0xE0) | ((v & 0x7) << 5); break;
        case CSR_FCSR:   l->fcsr = v; break;
        default: break;
    }
}

/* ============================================================
 * FP context sync
 * ============================================================ */
#define FP_SYNC(l) do { \
    uint8_t _f = ((l)->fcsr >> 5) & 0x7; \
    (l)->fp_status.float_rounding_mode = (_f <= 4) ? _f : 0; \
    (l)->fp_status.float_exception_flags = float_flag_inexact; \
} while(0)

#define FP_BACK(l) do { \
    (l)->fcsr = ((l)->fcsr & ~0x1F) | ((l)->fp_status.float_exception_flags & 0x1F); \
} while(0)

#endif /* SIMD_HANDLERS_H */
