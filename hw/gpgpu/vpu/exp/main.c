/*
 * main.c — VPU Standalone CLI
 * Copyright (c) 2024-2025, GPL v2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "config.h"
#include "test_runner.h"

static void help(void) {
    printf("Usage: vpu [command] [options]\n\n");
    printf("Commands:\n");
    printf("  (none)            run all tests\n");
    printf("  func              functional verification only\n");
    printf("  bench             performance benchmarks only\n");
    printf("  native            native C comparison\n");
    printf("Options:\n");
    printf("  -c <config.lua>   use config file (default: gpu_config.lua)\n");
}

int main(int argc, char **argv) {
    GPGPUState s; memset(&s, 0, sizeof(s));
    s.warp_size = 32; s.global_status = 1;

    const char *cfg_path = "gpu_config.lua";
    const char *cmd = NULL, *filter = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i+1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "func"))   cmd = "func";
        else if (!strcmp(argv[i], "bench"))  cmd = "bench";
        else if (!strcmp(argv[i], "native")) cmd = "native";
        else if (!strcmp(argv[i], "help") || !strcmp(argv[i], "-h")) { help(); return 0; }
        else filter = argv[i];
    }

    vpu_config_load(&s.cfg, cfg_path);
    s.vram_size = s.cfg.vram_mb * 1024 * 1024;
    s.vram_ptr = calloc(1, s.vram_size);
    if (!s.vram_ptr) { fprintf(stderr, "VRAM alloc failed\n"); return 1; }

    printf("VRAM: %lu MB | %u CU x %u warps/CU x %u lanes/warp\n\n",
           (unsigned long)(s.cfg.vram_mb), s.cfg.num_cus, s.cfg.warps_per_cu, s.warp_size);

    int errors = 0;
    if (cmd && !strcmp(cmd, "native"))
        test_run_native(&s);
    else if (cmd)
        errors = test_run(&s, cmd, filter);
    else
        test_run_all(&s);

    free(s.vram_ptr);
    return errors ? 1 : 0;
}
