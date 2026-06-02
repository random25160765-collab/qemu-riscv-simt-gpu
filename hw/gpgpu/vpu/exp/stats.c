/*
 * stats.c — VPU Statistics Collection & Display
 * Copyright (c) 2024-2025, GPL v2
 */
#include <stdio.h>
#include <string.h>
#include "state.h"
#include "stats.h"
#include "../core/utils.h"

/* color shortcuts */
#define KNRM COLOR_RESET
#define KBLD STYLE_BOLD
#define KDIM STYLE_DIM
#define KRED COLOR_RED
#define KGRN COLOR_GREEN
#define KYEL COLOR_YELLOW
#define KBLU COLOR_BLUE
#define KMAG COLOR_MAGENTA
#define KCYN COLOR_CYAN

#define MAX_ENTRIES 128
static StatsEntry entries[MAX_ENTRIES];
static int n_entries = 0;

void stats_reset(void)
{
    n_entries = 0;
}

void stats_snapshot(const GPGPUState *s, const char *name, double us, uint64_t flops, bool bench, bool pass)
{
    if (n_entries >= MAX_ENTRIES) return;
    StatsEntry *e = &entries[n_entries++];
    e->name = name;
    e->us = us;
    e->flops = flops;
    e->warps = s->stats.total_warps;
    e->bytes_r = s->stats.bytes_read;
    e->bytes_w = s->stats.bytes_write;
    memcpy(e->cat, s->stats.cat, sizeof(e->cat));
    memcpy(e->cat_static, s->stats.cat_static, sizeof(e->cat_static));
    e->branches = s->stats.total_branches;
    e->diverges = s->stats.simt_diverges;
    e->cache_hits = s->stats.cache_hits;
    e->cache_misses = s->stats.cache_misses;
    e->coal_ops = s->stats.coal_ops;
    e->coal_total = s->stats.coal_total;
    e->bench = bench;
    e->pass = pass;
}

/* --- formatting helpers (caller provides buffer) --- */
static char *fmt_n(char *buf, size_t sz, double v)
{
    if (v < 1024)
        snprintf(buf, sz, "%.0f", v);
    else if (v < 1024 * 1024)
        snprintf(buf, sz, "%.1fK", v / 1024);
    else
        snprintf(buf, sz, "%.1fM", v / 1e6);
    return buf;
}

static char *fmt_mix(char *buf, size_t sz, const uint64_t cat[4])
{
    uint64_t tot = 0;
    for (int i = 0; i < 4; i++)
        tot += cat[i];
    if (tot == 0) {
        buf[0] = '\0';
        return buf;
    }
    static const char *cn[] = {"ALU", "FP ", "MEM", "BR "};
    /* fixed: 4×(4+3) = 28 chars, always all four */
    snprintf(buf, sz, "%s%2.0f%% %s%2.0f%% %s%2.0f%% %s%2.0f%%", cn[0], 100.0 * cat[0] / tot, cn[1],
             100.0 * cat[1] / tot, cn[2], 100.0 * cat[2] / tot, cn[3], 100.0 * cat[3] / tot);
    return buf;
}

/* --- render --- */
void stats_render(void)
{
    if (n_entries == 0) return;

    /* detect which columns have data */
    bool has_perf = false, has_bw = false, has_mix = false, has_div = false;
    bool has_cache = false, has_coal = false;
    for (int i = 0; i < n_entries; i++) {
        StatsEntry *e = &entries[i];
        if (e->bench && e->flops) has_perf = true;
        if (e->bytes_r + e->bytes_w > 0) has_bw = true;
        uint64_t mt = 0;
        for (int c = 0; c < 4; c++)
            mt += e->cat[c] + e->cat_static[c];
        if (mt > 0) has_mix = true;
        if (e->branches > 0) has_div = true;
        if (e->cache_hits + e->cache_misses > 0) has_cache = true;
        if (e->coal_total > 0) has_coal = true;
    }

    int name_w = 5;
    for (int i = 0; i < n_entries; i++) {
        int w = (int)strlen(entries[i].name);
        if (w > name_w) name_w = w;
    }

    /* header */
    printf("  " KBLD "%-*s %8s" KNRM, name_w, "name", "time");
    if (has_perf) printf(" %6s", "perf");
    printf(" %6s", "warps");
    if (has_bw) printf("  %-26s", "bandwidth");
    if (has_mix) printf("  %-28s", "mix");
    if (has_div) printf(" %-12s", "div");
    if (has_cache) printf(" %7s", "cache");
    if (has_coal) printf(" %7s", "coalesc");
    printf(" %s\n", "");

    printf("  " KDIM "%-*s %8s", name_w, "----", "------");
    if (has_perf) printf(" %6s", "-----");
    printf(" %6s", "-----");
    if (has_bw) printf("  %-26s", "--------------------------");
    if (has_mix) printf("  %-28s", "----------------------------");
    if (has_div) printf(" %-12s", "------------");
    if (has_cache) printf(" %7s", "-------");
    if (has_coal) printf(" %7s", "-------");
    printf(" %s\n" KNRM, "----");

    /* rows */
    for (int i = 0; i < n_entries; i++) {
        StatsEntry *e = &entries[i];
        printf("  " KBLD "%-*s" KNRM " %7.0fus", name_w, e->name, e->us);
        if (has_perf) {
            if (e->bench && e->flops && e->us > 0)
                printf(" " KMAG "%5.0fM" KNRM, e->flops / e->us);
            else
                printf(" %6s", "");
        }
        printf(" %5luw", (unsigned long)e->warps);

        if (has_bw) {
            if (e->bytes_r + e->bytes_w > 0) {
                char rn[16], wn[16];
                double b = (e->bytes_r + e->bytes_w) / e->us;
                printf("  " KBLU "r:" KNRM "%s " KBLU "w:" KNRM "%s %6.0fM/s", fmt_n(rn, sizeof(rn), e->bytes_r),
                       fmt_n(wn, sizeof(wn), e->bytes_w), b);
            } else {
                printf("  %26s", "");
            }
        }

        if (has_mix) {
            char mx[64], tag[16];
            uint64_t td = e->cat[0] + e->cat[1] + e->cat[2] + e->cat[3];
            bool is_dyn = td > 0;
            fmt_mix(mx, sizeof(mx), is_dyn ? e->cat : e->cat_static);
            snprintf(tag, sizeof(tag), "%s%s", is_dyn ? KCYN "d" KNRM : KDIM "s" KNRM, is_dyn ? "yn " : "t ");
            printf("  %s%s", tag, mx);
        }

        if (has_div) {
            if (e->branches > 0 && e->diverges > 0)
                printf(" " KRED "%lu/%lu(%.0f%%)" KNRM, (unsigned long)e->diverges, (unsigned long)e->branches,
                       100.0 * e->diverges / e->branches);
            else
                printf("  %8s", "");
        }

        if (has_cache) {
            uint64_t tot = e->cache_hits + e->cache_misses;
            if (tot > 0)
                printf(" %6.0f%%", 100.0 * e->cache_hits / tot);
            else
                printf(" %7s", "");
        }

        if (has_coal) {
            if (e->coal_total > 0)
                printf(" %6.0f%%", 100.0 * e->coal_ops / e->coal_total);
            else
                printf(" %7s", "");
        }

        printf("  %s%s%s\n", e->pass ? KGRN : KRED, e->pass ? " PASS" : " FAIL", KNRM);
    }
}
