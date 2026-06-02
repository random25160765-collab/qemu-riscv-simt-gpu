/*
 * config.h — Lua-based VPU configuration
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef VPU_CONFIG_H
#define VPU_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    bool vpu, tcu, sfu, lp, debug, trace, perf;
} vpu_features_t;

typedef struct {
    uint32_t num_cus;
    uint32_t warps_per_cu;
    uint32_t warp_size;
    uint64_t vram_mb;
    vpu_features_t features;
} vpu_config_t;

void vpu_config_default(vpu_config_t *c);
int vpu_config_load(vpu_config_t *c, const char *path);

#endif /* VPU_CONFIG_H */
