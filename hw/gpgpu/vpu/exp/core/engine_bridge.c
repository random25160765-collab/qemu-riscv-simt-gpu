/*
 * engine_bridge.c — CGo bridge: non-inline wrappers for Go scheduler.
 *
 * All thin wrappers. No new logic. Engine / predecode / soa untouched.
 */
#include <string.h>
#include "engine_bridge.h"
#include "soa.h"
#include "predecode.h"
#include "gpgpu_core.h"

/* ── predecode ─────────────────────────────────────────────── */
ThOp *bridge_predecode(GPGPUState *s, uint32_t kern_addr, uint32_t kern_size, int *out_count)
{
    SIMDDecoder dec = {0};
    simd_decoder_init(&dec);
    ThOp *code = simd_predecode(&dec, s, kern_addr, kern_size, out_count);
    if (!code) return NULL;
    engine_resolve_handlers(code, *out_count);
    return code;
}

/* ── AoS ↔ SoA ─────────────────────────────────────────────── */
void bridge_aos_to_soa(const GPGPUWarp *warp, uint32_t *gpr, uint32_t *fpr, uint32_t *vpr, uint32_t *pc,
                       uint32_t *mhartid, uint32_t *fcsr)
{
    aos_to_soa(warp, gpr, fpr, vpr, pc, mhartid, fcsr);
}

void bridge_soa_to_aos(GPGPUWarp *warp, const uint32_t *gpr, const uint32_t *fpr, const uint32_t *vpr,
                       const uint32_t *pc, const uint32_t *mhartid, const uint32_t *fcsr)
{
    soa_to_aos(warp, gpr, fpr, vpr, pc, mhartid, fcsr);
}

/* ── sizeof helpers (for Go-side C.calloc) ────────────────── */
size_t bridge_sizeof_gpgpu_state(void)
{
    return sizeof(GPGPUState);
}
size_t bridge_sizeof_gpgpu_warp(void)
{
    return sizeof(GPGPUWarp);
}

/* ── VRAM allocator (delegates to vram_alloc.c) ───────────── */
#include "vram_alloc.h"
void bridge_vram_alloc_init(struct GPGPUState *s)
{
    vram_alloc_init(s);
}
void bridge_vram_alloc_reset(struct GPGPUState *s)
{
    vram_alloc_reset(s);
}
uint32_t bridge_vram_alloc(struct GPGPUState *s, size_t size)
{
    return vram_alloc(s, size);
}
uint32_t bridge_vram_ptr_read(struct GPGPUState *s, int slot)
{
    return vram_ptr_read(s, slot);
}
void bridge_vram_ptr_write(struct GPGPUState *s, int slot, uint32_t addr)
{
    vram_ptr_write(s, slot, addr);
}

/* ── warp init ─────────────────────────────────────────────── */
void bridge_init_warp(GPGPUWarp *warp, uint32_t kern_addr, uint32_t tid_base, const uint32_t block_id[3],
                      uint32_t n_threads, uint32_t warp_id, uint32_t blk_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = tid_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    if (n_threads >= GPGPU_WARP_SIZE)
        warp->active_mask = 0xFFFFFFFF;
    else
        warp->active_mask = (1U << n_threads) - 1;
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *l = &warp->lanes[i];
        l->pc = kern_addr;
        l->mhartid = MHARTID_ENCODE(blk_linear, warp_id, i);
        l->active = (warp->active_mask & (1 << i)) != 0;
        l->gpr[0].u32 = 0;
        l->fpr[0].u32 = 0;
    }
}
