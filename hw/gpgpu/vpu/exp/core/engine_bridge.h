/*
 * engine_bridge.h — CGo bridge: exports scheduler-critical functions
 * for Go's goroutine-based scheduler. All functions are non-inline
 * wrappers around existing static/inline implementations.
 */
#ifndef ENGINE_BRIDGE_H
#define ENGINE_BRIDGE_H

#include <stdint.h>
#include "state.h"
#include "gpgpu_core.h"
#include "../inst/dispatch_list.h" /* 必须在 engine.h 之前, trie/dispatch 对齐 */
#include "engine.h"
#include "dispatch.h"

/* predecode: kernel binary → ThOp array (wraps static inline simd_predecode) */
ThOp *bridge_predecode(GPGPUState *s, uint32_t kern_addr, uint32_t kern_size, int *out_count);

/* AoS ↔ SoA marshalling (wraps static inline aos_to_soa / soa_to_aos) */
void bridge_aos_to_soa(const GPGPUWarp *warp, uint32_t *gpr, uint32_t *fpr, uint32_t *vpr, uint32_t *pc,
                       uint32_t *mhartid, uint32_t *fcsr);
void bridge_soa_to_aos(GPGPUWarp *warp, const uint32_t *gpr, const uint32_t *fpr, const uint32_t *vpr,
                       const uint32_t *pc, const uint32_t *mhartid, const uint32_t *fcsr);

/* warp init (wraps static scheduler_init_warp — avoids duplicating MHARTID_ENCODE) */
void bridge_init_warp(GPGPUWarp *warp, uint32_t kern_addr, uint32_t tid_base, const uint32_t block_id[3],
                      uint32_t n_threads, uint32_t warp_id, uint32_t blk_linear);

/* sizeof helpers for Go-side C.calloc */
size_t bridge_sizeof_gpgpu_state(void);
size_t bridge_sizeof_gpgpu_warp(void);

/* VRAM allocator (from vram_alloc.c) */
void bridge_vram_alloc_init(struct GPGPUState *s);
void bridge_vram_alloc_reset(struct GPGPUState *s);
uint32_t bridge_vram_alloc(struct GPGPUState *s, size_t size);
uint32_t bridge_vram_ptr_read(struct GPGPUState *s, int slot);
void bridge_vram_ptr_write(struct GPGPUState *s, int slot, uint32_t addr);

#endif /* ENGINE_BRIDGE_H */
