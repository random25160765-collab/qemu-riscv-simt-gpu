/*
 * stats.h — VPU Statistics Collection & Display
 * Copyright (c) 2024-2025, GPL v2
 */
#ifndef VPU_STATS_H
#define VPU_STATS_H

#include <stdint.h>
#include <stdbool.h>
#include "state.h"

/* one test result */
typedef struct {
    const char *name;
    double us;
    uint64_t flops;
    uint64_t warps;
    uint64_t bytes_r, bytes_w;
    uint64_t cat[10], cat_static[10];
    uint64_t branches, diverges;
    uint64_t cache_hits, cache_misses;
    uint64_t coal_ops, coal_total;
    bool pass, bench;
} StatsEntry;

/* collect: snapshots s->stats + metadata */
void stats_snapshot(const GPGPUState *s, const char *name, double us, uint64_t flops, bool bench, bool pass);

/* render table */
void stats_render(void);

/* reset for next group */
void stats_reset(void);

#endif
