/*
 * test_runner.c — VPU Test Framework
 * Copyright (c) 2024-2025, GPL v2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "state.h"
#include "scheduler.h"
#include "stats.h"
#include "test_runner.h"
#include "../core/utils.h"

#define KERN_ADDR 0x500000
#define MAX_TESTS 64
static TestCase tests[MAX_TESTS];
static int n_tests = 0;
void test_register(TestCase t)
{
    if (n_tests < MAX_TESTS) tests[n_tests++] = t;
}

/* --- helpers --- */
static uint8_t *read_file(const char *path, size_t *out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return NULL;
    }
    struct stat st;
    fstat(fd, &st);
    *out = st.st_size;
    uint8_t *buf = malloc(st.st_size);
    if (!buf) {
        close(fd);
        return NULL;
    }
    if (read(fd, buf, st.st_size) != (ssize_t)st.st_size) {
        perror("read");
        free(buf);
        close(fd);
        return NULL;
    }
    close(fd);
    return buf;
}
void load_kernel(const char *path, GPGPUState *s, uint32_t addr)
{
    size_t sz;
    uint8_t *buf = read_file(path, &sz);
    if (buf) {
        memcpy(s->vram_ptr + addr, buf, sz);
        free(buf);
        s->kern_size = (uint32_t)sz;
    }
}

static uint64_t g_bp[3]; /* bench params pass-through */

/* --- run one test --- */
static int run_one(GPGPUState *s, TestCase *t)
{
    memset(s->vram_ptr, 0, 5 * 1024 * 1024);
    memcpy(g_bp, t->params, sizeof(g_bp));
    if (t->setup) t->setup(s);
    s->kernel.kernel_addr = KERN_ADDR;
    memcpy(s->kernel.grid_dim, t->grid, sizeof(t->grid));
    memcpy(s->kernel.block_dim, t->block, sizeof(t->block));
    if (t->kernel) load_kernel(t->kernel, s, KERN_ADDR);
    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    int ret = scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    if (ret) return 1;
    double us = (T1.tv_sec - T0.tv_sec) * 1e6 + (T1.tv_nsec - T0.tv_nsec) * 1e-3;
    int err = t->check ? t->check(s) : 0;
    stats_snapshot(s, t->name, us, t->flops, t->bench, !err);
    return err ? 1 : 0;
}

/* extern: test case registration (defined in tests/ directory) */
extern void tests_register(void);

int test_run(GPGPUState *s, const char *group, const char *filter)
{
    tests_register();
    int errors = 0;
    stats_reset();
    printf(STYLE_BOLD "%s=== %s ===%s\n\n",
           group && !strcmp(group, "bench")  ? COLOR_YELLOW
           : group && !strcmp(group, "func") ? COLOR_CYAN
                                             : "",
           group ? group : "All", COLOR_RESET);
    for (int i = 0; i < n_tests; i++) {
        TestCase *t = &tests[i];
        if (filter && !strstr(t->name, filter)) continue;
        if (group && !strcmp(group, "func") && t->bench) continue;
        if (group && !strcmp(group, "bench") && !t->bench) continue;
        printf("--- %s ---\n", t->name);
        errors += run_one(s, t);
    }
    printf("\n");
    stats_render();
    printf("\n");
    return errors;
}

void test_run_all(GPGPUState *s)
{
    tests_register();
    int errors = 0;
    errors += test_run(s, "func", NULL);
    errors += test_run(s, "bench", NULL);
    test_run_native(s);
    printf("===========================================\n"
           "Total errors: %d\n"
           "===========================================\n",
           errors);
}

/* bench params accessor (used by tests/bench.c) */
uint64_t *test_bp(void)
{
    return g_bp;
}
