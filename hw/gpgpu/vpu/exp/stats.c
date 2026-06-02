/*
 * stats.c — VPU Statistics Collection & Display
 * Copyright (c) 2024-2025, GPL v2
 */
#include <stdio.h>
#include <string.h>
#include "state.h"
#include "stats.h"
#include "../core/utils.h"
#include "../core/table.h"
#include "memory.h" /* DLOG */

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
    memcpy(e->cat, s->stats.cat, 10 * sizeof(uint64_t));
    memcpy(e->cat_static, s->stats.cat_static, 10 * sizeof(uint64_t));
    e->branches = s->stats.total_branches;
    e->diverges = s->stats.simt_diverges;
    e->cache_hits = s->stats.cache_hits;
    e->cache_misses = s->stats.cache_misses;
    e->coal_ops = s->stats.coal_ops;
    e->coal_total = s->stats.coal_total;
    e->bench = bench;
    e->pass = pass;
    DLOG(s, "[debug] snapshot %s: us=%.0f warps=%lu r=%lu w=%lu\n", name, us, (unsigned long)s->stats.total_warps,
         (unsigned long)s->stats.bytes_read, (unsigned long)s->stats.bytes_write);
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

static char *fmt_mix(char *buf, size_t sz, const uint64_t cat[10])
{
    static const char *cn[10] = {"ALU", "FP ", "MEM", "BR ", "SYS", "VPU", "TCU", "LP ", "SCI", "?"};
    uint64_t tot = 0;
    for (int i = 0; i < 10; i++)
        tot += cat[i];
    if (tot == 0) {
        buf[0] = '\0';
        return buf;
    }
    int off = 0;
    /* show ALU/FP/MEM/BR always, others only if >0 */
    for (int i = 0; i < 10; i++)
        if (cat[i] || i < 4) off += snprintf(buf + off, sz - off, "%s%2.0f%% ", cn[i], 100.0 * cat[i] / tot);
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
        for (int c = 0; c < 10; c++)
            mt += e->cat[c] + e->cat_static[c];
        if (mt > 0) has_mix = true;
        if (e->branches > 0) has_div = true;
        if (e->cache_hits + e->cache_misses > 0) has_cache = true;
        if (e->coal_total > 0) has_coal = true;
    }

    /* build column definitions */
    ColDef cols[TABLE_MAX_COLS];
    int nc = 0;
    cols[nc++] = (ColDef){.header = "name", .width = 0, .right = false, .color = -1};
    cols[nc++] = (ColDef){.header = "time", .width = 0, .right = true, .color = -1};
    if (has_perf) cols[nc++] = (ColDef){.header = "perf", .width = 0, .right = true, .color = -1};
    cols[nc++] = (ColDef){.header = "warps", .width = 0, .right = true, .color = -1};
    if (has_bw) cols[nc++] = (ColDef){.header = "bandwidth", .width = 0, .right = false, .color = -1};
    if (has_mix) cols[nc++] = (ColDef){.header = "mix", .width = 0, .right = false, .color = -1};
    if (has_div) cols[nc++] = (ColDef){.header = "div", .width = 0, .right = false, .color = -1};
    if (has_cache) cols[nc++] = (ColDef){.header = "cache", .width = 0, .right = true, .color = -1};
    if (has_coal) cols[nc++] = (ColDef){.header = "coalesc", .width = 0, .right = true, .color = -1};
    cols[nc++] = (ColDef){.header = "", .width = 0, .right = false, .color = -1};

    Table t;
    table_init(&t, cols, nc);

    for (int i = 0; i < n_entries; i++) {
        StatsEntry *e = &entries[i];
        char *row[TABLE_MAX_COLS];
        char tmp[16][64]; /* scratch buffers */
        int bi = 0;       /* buffer index */
        int ci = 0;

        row[ci++] = (char *)e->name;

        snprintf(tmp[bi], sizeof(tmp[bi]), "%.0fus", e->us);
        row[ci++] = tmp[bi];
        bi++;

        if (has_perf) {
            if (e->bench && e->flops && e->us > 0)
                snprintf(tmp[bi], sizeof(tmp[bi]), KMAG "%.0fM" KNRM, e->flops / e->us);
            else
                tmp[bi][0] = '\0';
            row[ci++] = tmp[bi];
            bi++;
        }

        snprintf(tmp[bi], sizeof(tmp[bi]), "%luw", (unsigned long)e->warps);
        row[ci++] = tmp[bi];
        bi++;

        if (has_bw) {
            if (e->bytes_r + e->bytes_w > 0) {
                char rn[16], wn[16];
                double b = (e->bytes_r + e->bytes_w) / e->us;
                snprintf(tmp[bi], sizeof(tmp[bi]), KBLU "r:" KNRM "%s " KBLU "w:" KNRM "%s %.0fM/s",
                         fmt_n(rn, sizeof(rn), e->bytes_r), fmt_n(wn, sizeof(wn), e->bytes_w), b);
            } else
                tmp[bi][0] = '\0';
            row[ci++] = tmp[bi];
            bi++;
        }

        if (has_mix) {
            char mx[128];
            uint64_t td = 0;
            for (int c = 0; c < 10; c++)
                td += e->cat[c];
            fmt_mix(mx, sizeof(mx), td ? e->cat : e->cat_static);
            snprintf(tmp[bi], sizeof(tmp[bi]), "%s%s %s", td ? KCYN "dyn" KNRM : KDIM "static" KNRM, "", mx);
            row[ci++] = tmp[bi];
            bi++;
        }

        if (has_div) {
            if (e->branches > 0 && e->diverges > 0)
                snprintf(tmp[bi], sizeof(tmp[bi]), KRED "%lu/%lu(%.0f%%)" KNRM, (unsigned long)e->diverges,
                         (unsigned long)e->branches, 100.0 * e->diverges / e->branches);
            else
                tmp[bi][0] = '\0';
            row[ci++] = tmp[bi];
            bi++;
        }

        if (has_cache) {
            uint64_t tot = e->cache_hits + e->cache_misses;
            if (tot > 0)
                snprintf(tmp[bi], sizeof(tmp[bi]), "%.0f%%", 100.0 * e->cache_hits / tot);
            else
                tmp[bi][0] = '\0';
            row[ci++] = tmp[bi];
            bi++;
        }

        if (has_coal) {
            if (e->coal_total > 0)
                snprintf(tmp[bi], sizeof(tmp[bi]), "%.0f%%", 100.0 * e->coal_ops / e->coal_total);
            else
                tmp[bi][0] = '\0';
            row[ci++] = tmp[bi];
            bi++;
        }

        row[ci++] = e->pass ? KGRN "PASS" KNRM : KRED "FAIL" KNRM;
        table_row(&t, (const char **)row);
    }

    table_render(&t);
}
