/*
 * main.c — VPU Interpreter Standalone Test Entry Point
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "state.h"
#include "test_runner.h"

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* 初始化设备状态 */
    GPGPUState s;
    memset(&s, 0, sizeof(s));
    s.num_cus      = 1;
    s.warps_per_cu = 1;
    s.warp_size    = 32;
    s.vram_size    = 64 * 1024 * 1024; /* 64 MB */
    s.global_status = 1; /* READY */

    s.vram_ptr = calloc(1, s.vram_size);
    if (!s.vram_ptr) {
        fprintf(stderr, "FATAL: failed to allocate VRAM (%lu MB)\n",
                (unsigned long)(s.vram_size >> 20));
        return 1;
    }
    printf("VRAM: %lu MB | Config: %u CU x %u warps/CU x %u lanes/warp\n\n",
           (unsigned long)(s.vram_size >> 20),
           s.num_cus, s.warps_per_cu, s.warp_size);

    /* 运行测试 */
    int errors = test_runner_run(&s);

    free(s.vram_ptr);
    return errors ? 1 : 0;
}
